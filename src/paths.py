from pathlib import Path


class ProjectPaths:
    """
    Centralized definition of key directory paths used throughout the project.
    All paths are resolved relative to the project root.
    """
    PROJECT_PATH = Path(__file__).resolve().parents[1]

    DATA_FOLDER = PROJECT_PATH / "data"
    DATA_RAW_FOLDER = DATA_FOLDER / "raw"

    REPORTS_FOLDER = PROJECT_PATH / "reports"
    REPORTS_FIGURES_FOLDER = REPORTS_FOLDER / "figures"

    NOTEBOOKS_FOLDER = PROJECT_PATH / "notebooks"

    @classmethod
    def get_all_paths(cls):
        return {
            name: value
            for name, value in vars(cls).items()
            if isinstance(value, Path)
        }


def establish_project_structure():
    """
    Creates all necessary directories defined in ProjectPaths.
    """
    for path_name, path in ProjectPaths.get_all_paths().items():
        if not path.exists():
            print("Establishing project structure...")
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {path_name} → {path}")


establish_project_structure()
