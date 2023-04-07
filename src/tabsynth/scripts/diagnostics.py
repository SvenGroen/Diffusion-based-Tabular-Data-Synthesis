import os
import sys
import site
import subprocess
import platform
import pkg_resources
def print_system_info():
    print("System Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"System: {platform.system()} {platform.release()} {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print()

def print_current_working_directory():
    cwd = os.getcwd()
    print("Current Working Directory:")
    print(cwd)
    print()

    print("Folders and Files in Current Working Directory:")
    for entry in os.listdir(cwd):
        entry_path = os.path.join(cwd, entry)
        if os.path.isfile(entry_path):
            print(f"File: {entry}")
        elif os.path.isdir(entry_path):
            print(f"Folder: {entry}")
    print()

def print_installed_packages():
    print("Installed Pip Packages:")
    installed_packages = sorted([(d.project_name, d.version) for d in pkg_resources.working_set], key=lambda x: x[0].lower())
    for package_name, package_version in installed_packages:
        print(f"{package_name}=={package_version}")
    print()

def get_python_interpreter_path():
    if sys.platform == "win32":
        cmd = ["where", "python"]
    else:
        cmd = ["which", "python"]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()

def get_installed_packages():
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()

def main():
    print("Python interpreter path:", get_python_interpreter_path())
    print_system_info()
    print_current_working_directory()
    print_installed_packages()
    print("sys.path:", sys.path)
    print("PYTHONPATH environment variable:", os.environ.get("PYTHONPATH", "Not set"))

    try:
        import tabsynth
        print("tabsynth imported successfully.")
    except ImportError as e:
        print("Error importing tabsynth:", e)

    print("Site-packages directory:", site.getsitepackages()[0])

    print("Installed packages:")
    print(get_installed_packages())


if __name__ == "__main__":
    main()
