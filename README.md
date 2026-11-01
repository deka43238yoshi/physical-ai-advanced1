# SO-101 サイコロ操作シミュレーション

MuJoCo上のSO-101で、**サイコロをつかんで持ち上げ、指定した数字の面を上にして置く**ことを目指す実験プロジェクトです。

WindowsのDocker DesktopでLinuxコンテナを使います。ホストのPython環境に依存せず、シミュレーション・学習データ・モデル・出力をローカルで扱います。

> 開発・検証段階です。限定条件の教師制御は成功していますが、**SmolVLAによる面1から面2への配置デモは未達**です。任意の初期条件や実機での成功・安全性を保証するものではありません。

## 現在の到達点

| 機能 | 状況 |
|---|---|
| テーブル・サイコロのヘッドレス物理計算と静止判定 | 確認済み |
| SO-101モデルの導入・関節駆動・接触を使った操作 | 限定条件で確認済み |
| 教師制御による上面1から面1・2・3への把持・配置 | 固定条件で確認済み。面1は上面を維持 |
| 観測した初期XY位置に基づく教師制御 | 各軸±5 mm、初期上面1・方位固定の限定範囲で検証 |
| 保存した成功・失敗軌道の動画・グラフ出力 | 確認済み。描画による再生であり、物理再実行ではない |
| SmolVLAのローカル学習・MuJoCo推論 | 配管を確認。面2タスクの物理成功は未達 |
| 面4〜6、任意初期位置・姿勢、実機操作 | 未対応または未検証 |
| 対話式GUI | 未検証 |

**教師制御と学習方策は別の方式です。** 教師制御はMuJoCoから位置・姿勢を取得し、IKと既知の動作手順を使います。SmolVLAはカメラ画像・関節状態・タスク指示から関節指令を予測します。教師が成功しても、学習方策の成功を意味しません。

## 設計方針

- タスク実行中のサイコロ位置・姿勢は直接書き換えず、ロボットとの物理接触で変化させます。
- 初期化、計画用の独立データ、保存軌道の描画用データへの状態設定は、物理タスク実行とは区別します。
- ヘッドレス実行を基本とし、可視化とGPU学習は別サービスに分けます。
- 接触・関節/力制限・支持安定などを検査し、不合格と実行エラーを明示します。
- 結果は実行ごとの新規フォルダーへ保存し、既存結果を上書きしません。
- データ・モデルの自動アップロードは行いません。モデル取得後はネットワーク無効の学習サービスを使用できます。

## 必要環境

- Windows、Docker Desktop、Linuxコンテナ（WSL 2バックエンド）
- Docker Composeプラグイン
- 初回ビルド時のコンテナイメージ・Python依存パッケージ取得のためのネットワーク

**ホストへのPython・pip・MuJoCoのインストールは不要です。** 通常の物理シミュレーションにGPUやXサーバーは不要です。可視化はOSMesaによるCPU描画です。

GPU学習は任意機能で、別途NVIDIA GPUとDockerからのGPU利用が必要です。ローカル検証環境ではRTX 4060 Laptop GPUを使用しました。他環境での動作を保証するものではありません。

## 最短の実行手順

ソースを取得したフォルダーのルートで、PowerShellから実行します。Docker Desktopを先に起動してください。

```powershell
docker version
docker compose version
docker info --format '{{.OSType}}'
docker compose config --quiet
docker compose build sim test
docker compose run --rm sim
```

`docker info`が`linux`を返すことを確認します。接続できない場合はDocker Desktopのエンジン起動・Linuxコンテナモード・WSL 2設定を確認してください。ホストPythonによる代用は想定していません。

既定の`sim`は**テーブルとサイコロの最小静止検証**です。SO-101の操作ではありません。

### SO-101と教師制御

```powershell
# SO-101単体の駆動を検証
docker compose run --rm sim --check-so101

# 初期上面1から、物理接触で持ち上げ、上面2で配置
docker compose run --rm sim --check-place --target-face 2
```

配置モードでは目標面1・2・3に対応しています。面ごとに既知の手順を使い、自由な障害物回避や任意姿勢への自律計画ではありません。

### 出力と終了コード

コンテナの`/app/outputs/<実行ID>`は、ホストの`.\outputs\<実行ID>`に対応します。JSONに判定・設定・失敗理由、CSVに状態・指令の時系列を保存します。`outputs`は公開対象に含めません。

| 終了コード | 意味 |
|---|---|
| 0 | 検証成功・結果保存完了 |
| 1 | 実行したが検証不合格 |
| 2 | 設定・読込・計算・入出力等のエラー |

実行直後の`$LASTEXITCODE`で確認できます。Docker起動・ビルドエラーは別扱いです。

### テスト

描画を含むテスト全体は、描画対応イメージで実行します。

```powershell
docker compose build visualize-test
docker compose run --rm --entrypoint python visualize-test -m pytest
```

軽量な`test`サービスにはPillow等の描画依存がないため、現状では全テストを実行すると比較可視化の3件が依存不足で失敗します。全体確認には上記の`visualize-test`を使ってください。ローカル学習成果物を環境変数で明示した場合だけ実行するintegration tests等はスキップされることがあります。スキップは検証成功を意味しません。

コード・設定・モデル資産・テストの変更後は、対応するイメージを再ビルドしてください。

## 任意機能

### 保存軌道の可視化

教師制御で配置を実行した後、その実行IDを指定します。

```powershell
docker compose build visualize
$runId = '<配置実行で表示された実行ID>'
docker compose run --rm visualize --run-dir "/app/outputs/$runId"
```

動画・静止画・グラフは`outputs\visualizations`へ保存します。これは保存状態の描画で、物理計算や成功判定の再検証ではありません。

### ローカル学習・推論・オフライン診断

`learning`サービスにはSmolVLA/LeRobot等の依存関係を、`learning-offline`にはネットワーク無効化とoffline設定を用意しています。

学習用教師データ、ダウンロード済みモデル、学習checkpoint、保存評価軌道は**同梱しません**。取得・生成・学習が必要なので、ソース取得だけで過去の実験を再現したり、学習済み方策を動かしたりできるわけではありません。

関連実装：

- [学習環境・モデル読込](src/so101_dice_sim/learning.py)
- [次周期末actionの教師データ](src/so101_dice_sim/endpoint_collection.py)
- [内部validationを固定した学習](src/so101_dice_sim/training_study.py)
- [学習位置での物理評価](src/so101_dice_sim/training_position_evaluation.py)
- [保存状態を使う推論候補比較](src/so101_dice_sim/inference_comparison.py)

関節actionは6関節のrad値、10 Hzの次周期末目標を500 Hzで線形補間する契約です。LeRobotの実機向けグリッパー値とそのまま交換できる単位ではありません。

## 現在検討している事項

学習データ上のaction誤差が改善しても、閉ループの物理タスク成功には至っていません。中央学習位置での直近評価は、把持前のテーブルと固定指の接触により停止しました。

保存軌道を使った診断では、初期の教師目標とのaction差と、その後の状態・指令の乖離が確認されています。同じnoiseで教師状態と方策状態を比べても指令差が残り、初期誤差は単一seedだけでは説明しにくい結果でした。

推論候補の固定オフライン比較では、積分回数増加・4回平均・ゼロnoiseを調べました。4回平均は教師状態での平均誤差を減らしましたが、保存方策状態の大きな乖離を解消していません。**推論設定の変更だけで面2デモが成功するとは確認できていません。**

今後は初期actionの適合、状態変化への応答、学習・推論条件の整合を検討します。追加データや物理再試行は、根拠と安全な検証条件を決めてから行います。内部validationによる方策選択と、未学習位置の最終評価は区別します。

## フォルダー構成

```text
src/so101_dice_sim/   Python実装・各実行入口
assets/              MJCF・SO-101モデル資産
config/              物理設定・学習依存関係
tests/               単体・物理・任意integration tests
tools/               モデル取得用ツール
outputs/             ローカル生成物（公開対象外）
Dockerfile           段階別のコンテナ環境
compose.yaml         実行・テスト・描画・学習サービス
```

## 制限と出典・ライセンス

SO-101はアーム5自由度とグリッパー1自由度です。任意の手先位置・姿勢を実現できるとは仮定しません。接触形状・駆動特性はシミュレーションモデルであり、実機の精密同定や安全認証ではありません。

- SO-101モデル：[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/robotstudio_so101)、元モデルは[The Robot Studio / SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)。
- 資産の[出典記録](assets/robotstudio_so101/SOURCE.md)・[ライセンス](assets/robotstudio_so101/LICENSE)・[SHA-256記録](assets/robotstudio_so101/provenance.json)を保持しています。上流資産はApache-2.0です。
- [MuJoCo](https://github.com/google-deepmind/mujoco)、[LeRobot](https://github.com/huggingface/lerobot)、[SmolVLA](https://huggingface.co/lerobot/smolvla_base)を利用します。依存ライブラリ・取得モデルにはそれぞれのライセンスが適用されます。
