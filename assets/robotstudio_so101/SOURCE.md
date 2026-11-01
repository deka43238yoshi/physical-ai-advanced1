# SO-101モデルの出典

- 配布元：[MuJoCo Menagerie / robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/robotstudio_so101)
- 固定コミット：`f3b6013629c7b94f8ee93fd977b161fa8350f9bc`
- 元モデル：[The Robot Studio / SO-ARM100 / Simulation/SO101](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)
- ライセンス：[Apache License 2.0](LICENSE)。上流のREADME・CHANGELOG・LICENSEを保持。
- 上流のMJCFと19件のSTLは無変更で保存。取得した23ファイルのSHA-256、上流Git blob SHAを [provenance.json](provenance.json) に記録。
- このSOURCE.mdとprovenance.jsonは本プロジェクトが追加した記録で、上流配布物ではない。
- 上流ルートのファイル一覧にはNOTICEなし（固定コミット時点）。

取得用のPowerShellスクリプトは [fetch_so101.ps1](../../tools/fetch_so101.ps1)。既存ディレクトリは上書きしないため、導入済みの通常利用では再取得不要。

本プロジェクトの [so101_scene.xml](../so101_scene.xml) は上流MJCFをincludeし、メッシュのパス、床、時間刻み0.002秒を別ファイルで設定する。上流MJCFを直接編集していない。

## 物理モデルに関する注意

- アーム5自由度と回転式グリッパー1自由度。任意の手先位置・姿勢を実現できるとは限らない。
- 指先の球・ボックス・カプセル・メッシュは接触形状の近似。描画形状と必ずしも一致しない。
- LeRobotのグリッパー0〜100と、このMJCFの回転角radは同じ値ではない。
- ゲインとトルク上限はシミュレーションモデルの値であり、実機の精密な同定結果として扱わない。
- 上流モデルにカメラ・マウントを含むが、今回カメラ認識・描画は使わない。マウントの質量・接触形状は保持している。
- 固定基部に衝突形状がないため、設置面や周辺物体との干渉は今後の統合で別途検討する。
