import argparse
import os

def add_tag(filename, tag, append):
    # Read the contents of the file
    with open(filename, "r") as f:
        contents = f.read()
        	    	
    # Replace spaces with commas	
    tag = ", ".join(tag.split())
    
    # Replace underscores with spaces
    tag = tag.replace("_", " ")
    
    # Check if the tag is already in the file
    if tag in contents:
        return  # Tag already exists, so do nothing
    
    # Add the tag to the contents
    if append:
        contents = contents.rstrip() + ", " + tag
    else:
        contents = tag + ", " + contents
    
    # Write the modified contents back to the file
    with open(filename, "w") as f:
        f.write(contents)

parser = argparse.ArgumentParser()
parser.add_argument("folder", help="the name of the folder containing the text files")
parser.add_argument("extension", help="the file extension to read (txt or caption)", choices=["txt", "caption"], default="txt")
parser.add_argument("tags", help="the tags to be added to the files, separated by spaces")
parser.add_argument("--append", help="append the tags to the end of the file (default) or to the beginning", action="store_true")
args = parser.parse_args()

folder = args.folder
extension = args.extension
tags = args.tags.split()
append = args.append

for filename in os.listdir(folder):
    if filename.endswith("." + extension):
        for tag in tags:
            add_tag(os.path.join(folder, filename), tag, append)
