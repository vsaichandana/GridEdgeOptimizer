import os

# Define the header content
header = """# -----------------------------------------------------------------------------
# GridEdgeOptimizer - A Comprehensive Grid Optimization Solution
# -----------------------------------------------------------------------------
#
# Version: 1.0
# Author: Sai Chandana Vallam Kondu, Iowa State University
# Copyright (c) 2023-2024
#
# This software is proprietary and confidential. Unauthorized copying,
# distribution, or use is strictly prohibited. All rights reserved by the author.
#
# Any use of this software requires explicit permission from the author and
# Iowa State University.
#
# This software is developed as a submodule to the next generation of the ISU-DERMS platform,
# which is supported by the U.S. Department of Energy (DOE) Cybersecurity, Energy Security,
# and Emergency Response (CESER) Award under Cooperative Agreement CR000016.

"""


def replace_or_append_header(file_path):
    """Replaces the header if it exists, otherwise appends it."""
    with open(file_path, "r+") as file:
        content = file.read()

        # If the header already exists, replace it
        if header in content:
            content = content.replace(header, header)
            file.seek(0)
            file.write(content)
            file.truncate()  # Truncate the file to remove any extra content at the end
            print(f"Header replaced in {file_path}")
        else:
            # If no header, prepend it to the file
            file.seek(0)
            file.write(header + "\n" + content)
            print(f"Header added to {file_path}")


def process_directory(directory):
    """Process all Python files in the given directory."""
    for dirpath, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".py")]:
            file_path = os.path.join(dirpath, filename)
            replace_or_append_header(file_path)


# Define the directory you want to process (change to your project's directory)
project_directory_src = "./src"
project_directory_test = "./tests"

if __name__ == "__main__":
    process_directory(project_directory_src)
    process_directory(project_directory_test)
