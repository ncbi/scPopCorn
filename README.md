# popcom
A python tool to do comparative analysis of mulitple single cell datasets.

# Install
```bash
$ pip install popcom
```

# Input File Format
popcom needs multiple single cell RNA-seq dataset as inputs. Bascially, the format look like the following. Example data file can be found in the ```data``` folder.

| Cell1ID | Cell2ID | Cell3ID | Cell4ID | Cell5ID  | ... |
|----|--------|--------|--------|---------|-----|
| Gene1 | 12 | 0 | 0 | 0 | ... |
| Gene2 | 125 | 0 | 298 | 0  | ... |
| Gene3 | 0 | 0| 0 | 0  | ... |
|...    |...|...|...|...|...|

# How to use

## import popcom package
```
from PopCom import MergeSingleCell
from PopCom import SingleCellData
```

## read in RNA-seq data
```
File1 = "../../Data/Human&Mouse_Kidney/GSE107585_Mouse_kidney_single_cell_seurat_data1.txt"
Test1 = SingleCellData()
Test1.ReadData_SeuratFormat(File1)


File3 = "../../Data/Human&Mouse_Kidney/GSE107585_Mouse_kidney_single_cell_seurat_data2.txt"
Test2 = SingleCellData()
Test2.ReadData_SeuratFormat(File3)

File5 = "../../Data/Human&Mouse_Kidney/GSE107585_Mouse_kidney_single_cell_seurat_data3.txt"
Test3 = SingleCellData()
Test3.ReadData_SeuratFormat(File5)


File5 = "../../Data/Human&Mouse_Kidney/GSE107585_Mouse_kidney_single_cell_seurat_data4.txt"
Test4 = SingleCellData()
Test4.ReadData_SeuratFormat(File5)

```
