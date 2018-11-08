# popcom
A python tool to do comparative analysis of mulitple single cell datasets.

# Install
```bash
$ pip install popcom
```

# Input File Format
popcom needs multiple single cell RNA-seq dataset as inputs. Bascially, the format look like this:
| Cell1ID | Cell2ID | Cell3ID | Cell4ID | Cell5ID  | ... |
|----|--------|--------|--------|---------|-----|
| Gene1 | 12 | 0 | 0 | 0 | ... |
| Gene2 | 125 | 0 | 298 | 0  | ... |
| Gene3 | 0 | 0| 0 | 0  | ... |

