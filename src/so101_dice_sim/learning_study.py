import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .endpoint_collection import collect_endpoints
from .grasp import run_grasp
from .reporting import create_run_directory, write_json
from .teacher_replay import action_table, replay_episode
from .visualization import load_replay


def split_positions() -> dict:
    train = [(x, y) for x in (-5., 0., 5.) for y in (-5., 0., 5.)]
    train += [(-3., 1.), (1., -3.), (3., 3.)]
    evaluation = [(-2.5, -2.5), (-2.5, 2.5), (2.5, -2.5), (2.5, 2.5)]
    return {
        "target_face": 2, "initial_orientation": "face1_up_fixed_heading",
        "train_offsets_mm": train, "evaluation_offsets_mm": evaluation,
        "policy": "評価位置は学習データ・正規化統計・更新選択に使用しない。失敗を除外して補充しない。",
    }


def prepare(directory: Path) -> dict:
    split = split_positions()
    write_json(directory / "split.json", split)
    results = []
    sources = []
    for group, positions in (
        ("train", split["train_offsets_mm"]), ("evaluation", split["evaluation_offsets_mm"]),
    ):
        for index, position in enumerate(positions):
            source = directory / f"{group}-{index:02d}-expert"
            source.mkdir()
            row = {"group": group, "index": index, "offset_mm": position}
            try:
                report = run_grasp(
                    source, place_face=2, observe_position=True,
                    initial_offset_m=tuple(value / 1000 for value in position),
                )
                if not report["validation"]["passed"]:
                    raise ValueError(f"元教師不合格: {report['validation']['failure_reasons']}")
                replay = load_replay(source)
                actions = action_table(replay)[np.arange(0, 22500, 50) + 1].astype(np.float32)
                output = directory / f"{group}-{index:02d}-endpoint"
                output.mkdir()
                result = replay_episode(output, source, replay, actions, "endpoint_linear_bounded")
                row.update({"passed": result["passed"], "source": str(output.resolve()),
                            "failure_reasons": result["failure_reasons"]})
                if group == "train" and result["passed"]:
                    sources.append(output)
            except (ValueError, RuntimeError, FloatingPointError) as error:
                row.update({"passed": False, "error": str(error)})
                write_json(source / "study_error.json", row)
            results.append(row)
            write_json(directory / f"progress-{len(results):02d}.json", {"split": split, "results": results})
            print(f"{group} {index + 1}/{len(positions)}: {row['passed']}", flush=True)
    if not all(row["passed"] for row in results):
        raise ValueError("教師検証に不合格あり。位置を入れ替えずデータ収集を停止します")
    collected = directory / "collection"
    collected.mkdir()
    collection = collect_endpoints(collected, sources)
    report = {
        "split": split, "results": results, "collection": str((collected / "collection.json").resolve()),
        "dataset": collection["dataset"], "uploaded": False,
    }
    write_json(directory / "study.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="面2の学習12本・評価4位置を固定し教師を物理検証・収集")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/learning/studies"))
    args = parser.parse_args(argv)
    directory = create_run_directory(args.output_dir)
    try:
        prepare(directory)
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        write_json(directory / "error.json", {"message": str(error)})
        print(f"学習準備エラー: {error}", file=sys.stderr)
        return 2
    print(f"学習準備保存先: {directory.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
