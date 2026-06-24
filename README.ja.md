# 日本風・架空島DEM生成器

[English](README.md) | **日本語**

実際のDEM(数値標高モデル)から日本の本土4島(本州・北海道・四国・九州。**南西諸島は除外**)
の地形を学習し、**現実の地形に統計的にも視覚的にも類似した架空の島の標高モデル**
(陸域 約5,000 km²、約60 m/px)を生成します。

生成は **2段ピクセル空間拡散カスケード** で行い、**キャンバス全体を継ぎ目処理なしの
単一フォワードパスで一括出力**します(オーバーラップ平均化・フェザリング・MultiDiffusion
は不使用)。データソースとアーキテクチャの根拠は `reports/DECISIONS.md`、先行研究調査は
`reports/research_synthesis.md`、最終指標は `reports/RESULTS.md`、変更履歴は
`reports/ITERATION_LOG.md` を参照。

**状況:** フェーズ0(PoC)・フェーズ1完了、さらに改善パス(Iter 7–8)実施済み。最終モデル
(**coarse@29k(起伏・陸率整合サンプリング)+ SR@12k**)は、約5,000 km²を中心とした
単一の連続した島(最高標高 ~2,860 m)を生成し、実地形と統計的に一致
(**標高KS 0.026、勾配KS 0.039、起伏KS 0.029** とほぼ完全、ハイプソ積分 0.17 vs 0.20、
径方向PSD傾き β 3.58 vs 実3.36、SWD 0.095)、視覚的にも自然な海岸線と樹枝状の山地水系
(火山錐状の地形を含む)を再現します。代表例:
`outputs/iter8/island_00_seed8_4991km2_shaded.png`。指標:`reports/RESULTS.md`。

## パイプライン

```
download_dem.py → preprocess.py → train.py (coarse) → train.py (sr) → generate.py → validate.py
   (GLO-30取得)    (LCC 60m         (EDM拡散・粗)      (EDM条件付きSR)   (カスケード   (統計+視覚)
                    モザイク+マスク)                                     +後選別)
```

- **coarse(粗・大局地形):** EDM拡散UNet、384²@240m、注意機構あり。島の輪郭・山脈・主要な谷を生成。
- **SR(詳細地形):** 注意機構なし・reflectパディングで**並進等変**なEDM条件付き拡散UNet、
  ×4で1536²@60mへ。256²パッチで学習し、生成時は**全キャンバスを1パスで**処理(タイリング無し)。
- **正規化:** `x = 2·√(clip(h,0,3776)/3776) − 1`(海面0→−1)。日本の右に偏った標高分布の
  低標高帯を拡張。指標は逆変換後の**メートル単位・陸ピクセルのみ**で算出。

## プロジェクト構成
```
data/raw/             Copernicus GLO-30 タイル(1°、GeoTIFF)        [一時]
data/processed/       japan_dem_60m.tif, japan_landmask_60m.tif, stats.json
src/                  download_dem, preprocess, dataset, networks, edm, train,
                      generate, validate, metrics, render, hydro
configs/              poc.yaml(フェーズ0), phase1.yaml(フェーズ1)
checkpoints/<名>/<段>/  latest.pt + ckpt_stepN.pt(EMA重みを内包)
outputs/              生成された島(.tif + hillshade/color/shaded PNG)
reports/              DECISIONS, RESULTS, ITERATION_LOG, research_synthesis, 比較図
scripts/              setup_env.sh, smoke_test.py, probe_throughput.py
```

## セットアップ
```bash
bash scripts/setup_env.sh        # uv venv (py3.11) + torch cu121 + 地理/ML ライブラリ
source .venv/bin/activate
```
主要バージョン:Python 3.11、torch 2.4.1+cu121、rasterio 1.3.10(GDAL 3.8.4)、
numpy 1.26、scipy 1.13。CUDAドライバ580 / RTX 2080 Ti(22.5 GB)。固定は `requirements.txt`。

## データ取得
```bash
python src/download_dem.py --out data/raw --workers 16     # 90タイル(約1.8 GB)
python src/preprocess.py   --raw data/raw --out data/processed
```
Copernicus DEM GLO-30(AWS Open Data、認証不要)から本土4島(南西諸島除外)を取得し、
日本中心のランベルト正角円錐図法で **60 m/px の継ぎ目なしモザイク**(int16メートル、海=0)、
陸マスク、`stats.json` を作成。学習時はこのモザイクから**オンザフライでクロップ**を抽出
(ディスク上にパッチデータセットを作らない)。再投影は terrain統計の正確性のため(4326の
緯度依存の非等方性を排除)。計測陸域 = 346,979 km²(実値とほぼ一致)。

## 学習(カスケード)
```bash
# フェーズ0 PoC(全経路を高速に検証)
python src/train.py --config configs/poc.yaml --stage coarse
python src/train.py --config configs/poc.yaml --stage sr

# フェーズ1(本番。バックグラウンド実行+ログ+定期サンプル出力)
python src/train.py --config configs/phase1.yaml --stage coarse   # --resume で再開
python src/train.py --config configs/phase1.yaml --stage sr
```
fp16 AMP + EMA + 勾配クリップ + 勾配チェックポイント(22.5GBに収める)。TensorBoardログは
`logs/<名>/<段>`。N ステップごとにチェックポイント保存(`--resume` で再開)。

## 生成
```bash
python src/generate.py --config configs/phase1.yaml \
  --coarse-ckpt checkpoints/phase1/coarse/latest.pt \
  --sr-ckpt     checkpoints/phase1/sr/latest.pt \
  --n 24 --keep 6 --target-km2 5000 --tol-km2 1000 --out outputs/generated
```
N個の粗候補を生成し、目標陸域(約5,000 km²)に近く**単一の連続した島**(最大連結成分で
後選別)を選び、SRで**全キャンバスを1パス**精緻化。GeoTIFF と hillshade/color/shaded PNG を出力。

## 2段階生成:1段階目だけで粗ドラフトを大量生成 → 一覧で選択 → 仕上げ
ステージ1(粗)は安価(約2〜3秒/枚)で、SR仕上げが高コストです。そこで、**`--coarse-only`
で1段階目だけを使ってラフを大量生成**(SRなし)し、一覧(contact_sheet.png)で吟味してから、
気に入ったものだけを `--complete` でフル解像度に仕上げられます。
```bash
# ステージ1 — 粗ドラフトを大量生成(SRなし):各ドラフトDEM(.npy)+プレビューPNG
#             + 一覧用 contact_sheet.png を保存。
python src/generate.py --config configs/phase1.yaml \
  --coarse-ckpt checkpoints/phase1/coarse/latest.pt \
  --coarse-only --n 100 --out outputs/drafts

# …outputs/drafts/contact_sheet.png を開いて、気に入ったID(例 003, 041)を控える…

# ステージ2 — 選んだドラフトをSRでフル1536pxに仕上げる(各1パス)。--hydro等も併用可。
python src/generate.py --config configs/phase1.yaml \
  --sr-ckpt checkpoints/phase1/sr/latest.pt \
  --complete outputs/drafts --pick 003 041 --out outputs/final --hydro
```
`--pick` はID部分文字列でマッチ(`003` / `coarse_003` / シード)。省略すると全ドラフトを完成。
ドラフトは正確な `.npy` DEM として保存されるため、選んだ地形が忠実に再現されます。
粗ドラフトは約2〜3秒/枚、SR仕上げは約2分/島。

## 水文学的整合性(任意)
```bash
python src/generate.py ... --hydro                  # 窪地を充填し海へ排水可能に
python src/generate.py ... --hydro --hydro-drainage # + D8河川網オーバーレイPNG
python src/generate.py ... --hydro --hydro-epsilon 0   # フラット充填(高速・平坦面は残る)
```
`--hydro` は各島を**水文学的に整合**させます:priority-flood で擬似的なシンクを充填し、
平坦面に微小勾配(`--hydro-epsilon`、既定1e-3 m)を与え、全陸地が海へ排水可能になります
(閉窪地なし・平坦面なし)。生の出力に対する典型的効果:陸の約10%のセルを嵩上げ
(平均約10 m)、厳密シンク約99%減、D8流量集積が約1.5k→約250kセル(現実的な樹枝状水系)。
`--hydro-drainage` は `*_drainage.png` 河川オーバーレイも出力。充填は約7秒/島。
実装は `src/hydro.py`(`fill_depressions`, `flow_accumulation`)。

## 検証(統計+視覚)
```bash
python src/validate.py --config configs/phase1.yaml \
  --gen-dir outputs/generated --n-real 32 --out reports/phase1
```
標高のKS検定/Wasserstein距離、勾配・起伏分布、ハイプソメトリック曲線、径方向パワースペクトル
傾き β とフラクタル次元 D、Sliced-Wasserstein距離(SWD)、陸域面積を算出し、実 vs 生成の
モンタージュと PSD/ハイプソ曲線を描画。視覚的な自己批評は、ヒルシェードPNGを開いて実地形の
レンダリングと並べて比較することで実施。

## ライセンス / 出典
- **コード:** MIT([LICENSE](LICENSE) 参照)。
- **Copernicus DEM GLO-30** — 出典明記により自由に利用可:
  *"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and
  Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all
  rights reserved."* 取得元:AWS Open Data バケット `copernicus-dem-30m`。
- 生成されたDEMは合成データであり、実在の場所ではありません。
```
