from PopCom import MergeSingleCell
from PopCom import SingleCellData

File1 = "./Human&Mouse_Pancreas/pancreas_human.expressionMatrix.txt"
File2 = "./Data/Human&Mouse_Pancreas/Supplementary_Table_PancreasCellData.tsv"
Test1 = SingleCellData()
Test1.ReadData_SeuratFormat(File1)
Test1.ReadTurth(File2, 0, 4, 3)

File1 = "./Data/Human&Mouse_Pancreas/pancreas_mouse.expressionMatrix.txt"
File2 = "./Data/Human&Mouse_Pancreas/Supplementary_Table_PancreasCellData.tsv"
Test2 = SingleCellData()
Test2.ReadData_SeuratFormat(File1)
Test2.ReadTurth(File2, 0, 4, 3)
