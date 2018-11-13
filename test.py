from popcom import MergeSingleCell
from popcom import SingleCellData

## read in RNA-seq data
File1 = "../../Data/Human&Mouse_Kidney/GSE107585_Mouse_kidney_single_cell_seurat_data1.txt"
Test1 = SingleCellData()
Test1.ReadData_SeuratFormat(File1)


File2 = "../../Data/Human&Mouse_Kidney/GSE107585_Mouse_kidney_single_cell_seurat_data2.txt"
Test2 = SingleCellData()
Test2.ReadData_SeuratFormat(File2)

File3 = "../../Data/Human&Mouse_Kidney/GSE107585_Mouse_kidney_single_cell_seurat_data3.txt"
Test3 = SingleCellData()
Test3.ReadData_SeuratFormat(File3)


File4 = "../../Data/Human&Mouse_Kidney/GSE107585_Mouse_kidney_single_cell_seurat_data4.txt"
Test4 = SingleCellData()
Test4.ReadData_SeuratFormat(File4)

## Normlize counts data, find highly vaiable genes, and natural logarithm of one plus of the counts data
Test1.Normalized_per_Cell()
Test1.FindHVG()
Test1.Log1P()

Test2.Normalized_per_Cell()
Test2.FindHVG()
Test2.Log1P()

Test3.Normalized_per_Cell()
Test3.FindHVG()
Test3.Log1P()

Test4.Normalized_per_Cell()
Test4.FindHVG()
Test4.Log1P()

## Combine data
MSingle = MergeSingleCell(Test1, Test2, Test3, Test4)

## Define supercells for each data sets
MSingle.MultiDefineSuperCell(200,200,200,200)

## Compute co-membership graph within each dataset and similarity matrix across dataset
MSingle.ConstructWithinSimiarlityMat_SuperCellLevel()
MSingle.ConstructBetweenSimiarlityMat_SuperCellLevel()

## Run joint partition
MSingle.SDP_NKcut(15, 3.0)

## Rounding the results
CResult = MSingle.NKcut_Rounding(3.0,20)

## Output the results
MSingle.OutputResult("TestOut.txt")
