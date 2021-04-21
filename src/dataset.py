import os

in_directory = os.fsencode("stories/")
    
for file in os.listdir(in_directory):
    filename = os.fsdecode(file)
    highlights = list()
    body = list()
    
    if filename.endswith(".story"): 
        with open(f"stories/{filename}", 'r', encoding='utf-8') as in_file:
            corpus_lines = in_file.readlines()

            highlight_line = False
            for line in corpus_lines:
                if line == "@highlight\n":
                    highlight_line = True
                    continue
                if highlight_line == True and len(line) > 1:
                    highlights.append(line)
                    highlight_line = False
                else:
                    body.append(line)

        with open(f"dataset/body/{filename}", 'w', encoding='utf-8') as out_file:
            for line in body:
                out_file.write(line)
            out_file.close()

        with open(f"dataset/highlights/{filename}", 'w', encoding='utf-8') as out_file:
            for line in highlights:
                out_file.write(line)
            out_file.close()
        