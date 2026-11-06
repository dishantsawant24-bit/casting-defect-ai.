# PyInstaller spec for a local Windows desktop bundle.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

root = Path.cwd()
model_path = root / "models" / "resnet18_best.pth"
if not model_path.is_file():
    raise FileNotFoundError("Place your locally held model checkpoint before building.")

datas = [(str(root / "app" / "app.py"), "app"), (str(root / "src"), "src"), (str(model_path), "models")]
datas += copy_metadata("streamlit")
datas += collect_data_files("cv2")

a = Analysis(
    [str(root / "launcher.py")], pathex=[str(root)], binaries=[], datas=datas,
    hiddenimports=["src.inference.predict", "src.inference.input_gate", "src.inference.app_storage", "streamlit.web.bootstrap", "cv2"],
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=["jupyter", "jupyterlab", "notebook", "ipykernel"], noarchive=False, optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="CastingInspectionAI",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name="CastingInspectionAI")
