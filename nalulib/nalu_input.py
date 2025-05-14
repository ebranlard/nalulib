""" Deal with nalu input file"""
import ruamel.yaml
from io import StringIO

class RuamelYamlEditor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.yaml = ruamel.yaml.YAML()
        self.yaml.preserve_quotes = True
        with open(filepath, 'r', encoding='utf-8') as f:
            f.seek(0)
            self.data = self.yaml.load(f)

    @property
    def lines(self):
        buf = StringIO()
        self.yaml.dump(self.data, buf)
        return buf.getvalue().splitlines(keepends=True)

    def save(self, outpath=None):
        with open(outpath or self.filepath, 'w', encoding='utf-8') as f:
            f.writelines(self.lines)

    def print(self, context=True, line=True):
        for i, l in enumerate(self.lines):
            c = ""
            if line:
                l = "{:50s}".format(l.rstrip())
            else:
                l = ""
            if context:
                c = '# ' + str(self.get_context(i))
            print(f"{i}: {l} {c}")


NALUInputFile = RuamelYamlEditor