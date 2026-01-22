import subprocess
import sys
from pathlib import Path


def generate_proto(proto_file=None, output_dir=None):
    script_dir = Path(__file__).parent.resolve()
    proto_file = Path(
        proto_file or script_dir / "models/semantic_graph.proto"
    ).resolve()
    output_dir = Path(output_dir or proto_file.parent).resolve()

    if not proto_file.exists():
        print(f"<error> Proto file not found: {proto_file}")
        return False

    try:
        subprocess.run(
            [
                "protoc",
                f"--proto_path={proto_file.parent}",
                f"--python_out={output_dir}",
                str(proto_file),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"<info> Generated: {output_dir / proto_file.stem}_pb2.py")
        return True
    except Exception as e:
        print(f"<error> Error: {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if generate_proto() else 1)
