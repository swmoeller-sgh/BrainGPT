import os
import shutil

# Define the project name and top-level directory
project_name = "my_project"
root_project_dir = os.getcwd()

# Define the names of the standard directories and files
dir_names = ["data", "docs", "models","notebooks","references","reports","src"]
file_names = ["README.md", "LICENSE", ".gitignore","Makefile", "requirements.txt",
              "setup.py"]

# Define the path to some standard files
little_helper = "/Users/swmoeller/python/[standard]_folder_structure/src/utils"
src_dir = "/Users/swmoeller/python/[standard]_folder_structure/src"
tree_structure_path = "/Users/swmoeller/python/[standard]_folder_structure/tree_structure.md"

# Define the names of subdirectories within the standard directories
data_subdirs = ["05_raw","10_external","15_interim","20_cleansed"]
models_subdirs = ["backup","trained_models"]
reports_subdirs = ["figures"]
src_subdirs = ["utils", "01_deployment","02_train","03_prediction","04_repository", "data","features","models",
               "visualization"]
code_dir = ["01_deployment","02_train","03_prediction"]

# Define the content to write to the README.md and LICENSE files
readme_content = f"# {project_name}\n\nThis is a placeholder README file for the {project_name} project."
license_content = "This project is licensed under the MIT License."

# Create the top-level directory if it doesn't already exist
"""if not os.path.exists(root_project_dir):
    os.mkdir(root_project_dir)
else:
    print(f"The directory {root_project_dir} already exists.")"""

# Copy the tree_structure.md file to the project root if it doesn't already exist
tree_structure_dest = os.path.join(root_project_dir, "tree_structure.md")
if not os.path.exists(tree_structure_dest):
    shutil.copy(tree_structure_path, tree_structure_dest)
else:
    print(f"The file {tree_structure_dest} already exists.")


# Create the standard directories if they don't already exist
for dir_name in dir_names:
    dir_path = os.path.join(root_project_dir, dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

        # Create subdirectories within the standard directories if applicable
        if dir_name == "data":
            for sub_dir_name in data_subdirs:
                sub_dir_path = os.path.join(dir_path, sub_dir_name)
                if not os.path.exists(sub_dir_path):
                    os.mkdir(os.path.join(dir_path, sub_dir_name))
        elif dir_name == "models":
            for sub_dir_name in models_subdirs:
                sub_dir_path = os.path.join(dir_path, sub_dir_name)
                if not os.path.exists(sub_dir_path):
                    os.mkdir(os.path.join(dir_path, sub_dir_name))
        elif dir_name == "report":
            for sub_dir_name in reports_subdirs:
                sub_dir_path = os.path.join(dir_path, sub_dir_name)
                if not os.path.exists(sub_dir_path):
                    os.mkdir(os.path.join(dir_path, sub_dir_name))
        elif dir_name == "src":
            for sub_dir_name in src_subdirs:
                sub_dir_path = os.path.join(dir_path, sub_dir_name)
                if not os.path.exists(sub_dir_path):
                    os.mkdir(os.path.join(dir_path, sub_dir_name))

                    # Copy all files from the source directory 
                        
                    if sub_dir_name in code_dir:
                        sub_code_dir = os.path.join(src_dir, sub_dir_name)
                        for file_name in os.listdir(sub_code_dir):
                            file_path = os.path.join(sub_code_dir, file_name)
                            shutil.copy(file_path, os.path.join(dir_path, sub_dir_name, file_name))
                    # Copy all files from the 00_little_helper directory to the src/utils/00_little_helper directory
                    if sub_dir_name == "utils":
                        helper_dir = "/Users/swmoeller/python/[standard]_folder_structure/src/utils"
                        for file_name in os.listdir(helper_dir):
                            file_path = os.path.join(helper_dir, file_name)
                            shutil.copy(file_path, os.path.join(dir_path, sub_dir_name, file_name))

    else:
        print(f"The directory {dir_path} already exists.")
        # Create subdirectories within the standard directories if applicable
        if dir_name == "data":
            for sub_dir_name in data_subdirs:
                sub_dir_path = os.path.join(dir_path, sub_dir_name)
                if not os.path.exists(sub_dir_path):
                    os.mkdir(os.path.join(dir_path, sub_dir_name))
        elif dir_name == "models":
            for sub_dir_name in models_subdirs:
                sub_dir_path = os.path.join(dir_path, sub_dir_name)
                if not os.path.exists(sub_dir_path):
                    os.mkdir(os.path.join(dir_path, sub_dir_name))
        elif dir_name == "report":
            for sub_dir_name in reports_subdirs:
                sub_dir_path = os.path.join(dir_path, sub_dir_name)
                if not os.path.exists(sub_dir_path):
                    os.mkdir(os.path.join(dir_path, sub_dir_name))
        elif dir_name == "src":
            for sub_dir_name in src_subdirs:
                sub_dir_path = os.path.join(dir_path, sub_dir_name)
                if not os.path.exists(sub_dir_path):
                    os.mkdir(os.path.join(dir_path, sub_dir_name))
                    # Copy all files from the utils directory to the src/utils directory
                    if sub_dir_name == "utils":
                        helper_dir = "/Users/swmoeller/python/[standard]_folder_structure/src/utils"
                        for file_name in os.listdir(helper_dir):
                            file_path = os.path.join(helper_dir, file_name)
                            shutil.copy(file_path, os.path.join(dir_path, sub_dir_name, file_name))

# Create the standard files and write content to them if they don't already exist
for file_name in file_names:
    file_path = os.path.join(root_project_dir, file_name)
    if not os.path.exists(file_path):
        open(file_path, "w").close()
        # Write content to the README.md file
        if file_name == "README.md":
            with open(file_path, "w") as readme_file:
                readme_file.write(readme_content)
        # Write content to the LICENSE file
        elif file_name == "LICENSE":
            with open(file_path, "w") as license_file:
                license_file.write(license_content)
    else:
        print(f"The file {file_path} already exists.")

print(f"Successfully created the standard directories and files for the {project_name} project.")
