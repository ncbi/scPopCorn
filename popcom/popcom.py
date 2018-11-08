# -*- coding: utf-8 -*-

"""Main module."""

import numpy as np
import pandas as pd
import anndata as ad
import operator
import matplotlib as mpl
mpl.use('TkAgg')
import scanpy.api as sc
from scanpy.api.pp import filter_genes_dispersion
from scanpy.api.pp import normalize_per_cell
from scanpy.api.pp import scale
from scanpy.api.pp import log1p
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scanpy.api.tl import pca
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from hub_toolbox import IO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import block_diag
import copy
import collections
import operator
import cvxpy as cvx
from scipy.sparse.csgraph import connected_components

def SDPMerge(LWithinbar, LBetweenbar, DWithindiag, DBetweendiag, K, Lambda):
    NumC = LWithinbar.shape[0]
    # create X
    X = cvx.Variable((NumC,NumC), PSD = True)
    # construct optimization
    Lbar = Lambda*LBetweenbar+LWithinbar
    Obj = cvx.Minimize(cvx.trace(X*Lbar))
    ConAll = [X == X.T, cvx.trace(X) == K,  X*(DBetweendiag) == (DBetweendiag),  X >= 0]#X*DCdiag == DCdiag,
    # solve
    prob = cvx.Problem(Obj, ConAll)
    prob.solve(solver=cvx.SCS)
    print("optimal value with SCS:", prob.value)
    X = X.value
    return X


class PageRankVec:
    def __init__(self, AdjMat, alpha, eps):
        self.AdjMat = AdjMat
        np.fill_diagonal(self.AdjMat, 0)
        self.alpha = alpha
        self.eps = eps
        self.Degree = np.sum(self.AdjMat, axis=0)
        self.NumVertex = AdjMat.shape[0]

    def Push(self, node_u, pvec, rvec):
        plvec = copy.deepcopy(pvec)
        rlvec = copy.deepcopy(rvec)
        plvec[node_u] = pvec[node_u] + self.alpha*rvec[node_u]
        rlvec[node_u] = (1.-self.alpha)*rvec[node_u]/2.
        Ind = np.nonzero(self.AdjMat[node_u,:])[0]
        for i in Ind:
            rlvec[i] = rvec[i] + (1.-self.alpha)*rvec[node_u]*self.AdjMat[node_u,i] / (2*self.Degree[node_u])
        return plvec, rlvec

    def ApproximatePR_Vertex(self, node_s):
        pvec = np.zeros(self.NumVertex)
        rvec = np.zeros(self.NumVertex)
        rvec[node_s] = 1.
        tmp_eps = self.eps
        tmp = rvec - tmp_eps*self.Degree
        Indtmp = np.where(tmp>0)[0]
        while np.sum(rvec) > 0.001:
            while Indtmp.size > 0:
                node_pick = Indtmp[0]
                pvec, rvec = self.Push(node_pick, pvec, rvec)
                tmp = rvec - tmp_eps*self.Degree
                Indtmp = np.where(tmp>0)[0]
            #print(np.count_nonzero(pvec))
            if np.count_nonzero(pvec) <= 1:
                tmp_eps = tmp_eps*0.8
                rvec[node_s] = 1.
                tmp = rvec - tmp_eps*self.Degree
                Indtmp = np.where(tmp>0)[0]
                #print(tmp_eps)
            else:
                break

        #print(pvec)
        #print(np.count_nonzero(pvec))
        #print(np.sum(pvec))
        #print(rvec)
        return pvec#, rvec

    def Conductance(self, setvec):
        setvec = setvec[np.newaxis]
        InnerEdge = setvec.dot(self.AdjMat).dot(setvec.T)
        VolS = np.sum(np.multiply(self.Degree, setvec))
        Vol = np.sum(self.Degree)

        if Vol-VolS < 1e-8:
            return 1.
        else:
            Cond = (VolS - InnerEdge) / np.min([VolS, Vol-VolS])
            return Cond
        #Cond = (VolS - InnerEdge) / VolS
        #return Cond

    def Density(self, setvec):
        setvec = setvec[np.newaxis]
        InnerEdge = setvec.dot(self.AdjMat).dot(setvec.T)
        Density = InnerEdge / (np.sum(setvec)**2)
        return Density

    def Sweep(self, pvec, Num):
        Ind = np.argsort(-pvec)[0:Num]
        CumInd = list()
        MinCond = 100000.
        for i in Ind:
            tmpvec = np.zeros(len(pvec))
            CumInd.extend([i])
            print(CumInd)
            tmpvec[CumInd] = 1
            Condtmp = self.Conductance(tmpvec)
            print(Condtmp)
            Dentmp = self.Density(tmpvec)
            print(Dentmp)
            if Condtmp < MinCond:
                MinCond = Condtmp
                goodvec = tmpvec
        return goodvec

    def Sweep_Density(self, pvec, Num):
        Ind = np.argsort(-pvec)[0:Num]
        CumInd = list()
        MaxDen = -100000.
        indicator = 0
        for i in range(len(Ind)):
            tmpvec = np.zeros(len(pvec))
            CumInd.extend([Ind[i]])
            #print(CumInd)
            tmpvec[CumInd] = 1
            #print(Condtmp)
            Dentmp = self.Density(tmpvec)
            Condtmp = self.Conductance(tmpvec)
            #print(Dentmp)
            if Dentmp /Condtmp >= MaxDen and i == indicator:
                MaxDen = Dentmp
                goodvec = tmpvec
                indicator = indicator + 1
            else:
                break
        return goodvec, MaxDen

    def KeepEdge(self, Num):
        KMat = np.zeros((self.NumVertex,self.NumVertex))
        for nodei in range(self.NumVertex):
            Pvec = self.ApproximatePR_Vertex(nodei)
            Pvec = np.divide(Pvec, self.Degree)
            kvec = self.Sweep(Pvec, Num)
            KMat[nodei,:] = KMat[nodei,:] + kvec
            KMat[:,nodei] = KMat[:,nodei] + kvec
        return KMat

    def ApproximatePR_Matrix(self):
        ProMat = np.zeros((self.NumVertex, self.NumVertex))
        for nodei in range(self.NumVertex):
            ProMat[nodei,:] = self.ApproximatePR_Vertex(nodei)
        return ProMat

    def ApproximatePR_Matrix_noDiag(self):
        ProMat = np.zeros((self.NumVertex, self.NumVertex))
        for nodei in range(self.NumVertex):
            #print("Node %d" % nodei)
            pvec = self.ApproximatePR_Vertex(nodei)
            pvec[nodei] = 0.
            pvec = pvec / np.sum(pvec)
            ProMat[nodei,:] = ProMat[nodei,:] + pvec
            ProMat[:, nodei] = ProMat[:, nodei] + pvec
        return ProMat

def shared_nearest_neighbors(D:np.ndarray, k:int=10, metric='similarity'):
    """Transform distance matrix using shared nearest neighbors [1]_.

    SNN similarity is based on computing the overlap between the `k` nearest
    neighbors of two objects. SNN approaches try to symmetrize nearest neighbor
    relations using only rank and not distance information [2]_.

    Parameters
    ----------
    D : np.ndarray
        The ``n x n`` symmetric distance (similarity) matrix.

    k : int, optional (default: 10)
        Neighborhood radius: The `k` nearest neighbors are used to calculate SNN.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether the matrix `D` is a distance or similarity matrix

    Returns
    -------
    D_snn : ndarray
        Secondary distance SNN matrix

    References
    ----------
    .. [1] R. Jarvis and E. A. Patrick, “Clustering using a similarity measure
           based on shared near neighbors,” IEEE Transactions on Computers,
           vol. 22, pp. 1025–1034, 1973.

    .. [2] Flexer, A., & Schnitzer, D. (2013). Can Shared Nearest Neighbors
           Reduce Hubness in High-Dimensional Spaces? 2013 IEEE 13th
           International Conference on Data Mining Workshops, 460–467.
           http://doi.org/10.1109/ICDMW.2013.101
    """
    IO._check_distance_matrix_shape(D)
    IO._check_valid_metric_parameter(metric)
    if metric == 'distance':
        self_value = 0.
        sort_order = 1
        exclude = np.inf
    if metric == 'similarity':
        self_value = 1.
        sort_order = -1
        exclude = -np.inf

    distance = D.copy()
    np.fill_diagonal(distance, exclude)
    n = np.shape(distance)[0]
    knn = np.zeros_like(distance, bool)

    # find nearest neighbors for each point
    for i in range(n):
        di = distance[i, :]
        nn = np.argsort(di)[::sort_order]
        knn[i, nn[0:k]] = True

    D_snn = np.zeros_like(distance)
    for i in range(n):
        knn_i = knn[i, :]
        j_idx = slice(i+1, n)

        # using broadcasting
        Dij = np.sum(np.logical_and(knn_i, knn[j_idx, :]), 1)
        if metric == 'distance':
            D_snn[i, j_idx] = 1. - Dij / k
        else: # metric == 'similarity':
            D_snn[i, j_idx] = Dij / k

    D_snn += D_snn.T
    np.fill_diagonal(D_snn, self_value)
    return D_snn

def nearest_neighbors(D:np.ndarray, k:int=10):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(D)
    D_knn = nbrs.kneighbors_graph(D).toarray()

    return D_knn


def SemiPostivifyMat(A):
    #U, S, Vh = np.linalg.svd(A, full_matrices=True)
    w, v = np.linalg.eig(A)
    w = w.real
    #v = v.real
    w[w<0.] = 0.
    #w.clip(min=0., out=w)
    #print(np.allclose(A, v.dot(np.diag(w)).dot(v.T)))
    #S.clip(min=0., out=S)
    A_SP = v.dot(np.diag(w)).dot(v.T).real#np.dot(U, np.dot(np.diag(S), Vh))
    return A_SP

def ComputeZ(X):
    M = X.shape[0]
    AllZeors = np.zeros((M, M))
    AllOnes = np.ones((M, M))
    Tmp1 = np.maximum(X, AllZeors)
    Tmp2 = np.minimum(Tmp1, AllOnes)
    np.fill_diagonal(Tmp2, 1)
    return Tmp2

def ADMM_Modulaity_SDP(PMat):
    M = PMat.shape[0]
    Zold = np.zeros((M,M))
    Lold = np.zeros((M,M))
    Objold = 0.
    for k in range(100):
        Ynew = SemiPostivifyMat(Zold-Lold+PMat)
        Znew = ComputeZ(Ynew + Lold)
        Lnew = Lold + Ynew - Znew

        Objnew = np.trace(Ynew.dot(PMat))
        print("Iter %d, Objnew %f Objold %f" % (k, Objnew, Objold))
        if k > 10 and Objold - Objnew > 0.:
            break
        else:
             Objold = Objnew

        Lold = Lnew
        Zold = Znew
    #Ynew = SemiPostivifyMat(Zold-Lold+PMat)
    return Ynew



class SingleCellData():

    def ReadData(self, FileName):
        fh = open(FileName, 'r')
        # get gene name
        tline = fh.readline()
        line = tline.strip().split(",")
        GeneName = [x.upper() for x in line[3:]]
        # get other info
        scRNAseqTmp = list()
        NumCell = 0
        CellID = list()
        ClusterLabel = list()
        for tline in fh:
            line = tline.strip().split(",")
            CellID.append(line[0])
            ClusterLabel.append(line[2])
            scRNAseqTmp.append([float(i) for i in line[3:]])
            NumCell = NumCell + 1
        obs = pd.DataFrame()
        obs['Cell_ID'] = CellID
        obs['Cell_Label'] = ClusterLabel
        var = pd.DataFrame(index = GeneName)
        self.scRNAseq_Counts = ad.AnnData(np.array(scRNAseqTmp), obs = obs, var = var)
        self.ClusterLabel = list(self.scRNAseq_Counts.obs['Cell_Label'])
        self.NumCell = NumCell
        fh.close()

    def Normalized_per_Cell(self):
        self.scRNAseq_Propcessed = normalize_per_cell(self.scRNAseq_Counts, copy= True)

    def Counts_To_Exp(self):
        CountsMat = self.scRNAseq_HVGData.X
        ExpMat = np.log1p(10000* ((CountsMat) / CountsMat.sum(axis=1)[:,np.newaxis]))
        self.scRNAseq_HVGData_log1p = ad.AnnData(ExpMat, obs = self.scRNAseq_HVGData.obs, var = self.scRNAseq_HVGData.var)

    def FindHVG(self):
        self.scRNAseq_HVGData = filter_genes_dispersion(self.scRNAseq_Propcessed, copy= True)

    def Log1P(self):
        self.scRNAseq_HVGData_log1p = log1p(self.scRNAseq_HVGData, copy=True)


    def ReadData_SeuratFormat(self, FileName):
        fh = open(FileName, 'r')
        # get cell id
        tline = fh.readline()
        line = tline.strip().split()
        CellID = [x for x in line]
        self.NumCell = len(CellID)
        print("Number of cells: %d" %self.NumCell)
        # get scRNA-seq data
        GeneName = list()
        scRNAseqTmp = list()
        for tline in fh:
            line = tline.strip().split()
            GeneName.append(line[0])
            scRNAseqTmp.append([float(i) for i in line[1:]])
        obs = pd.DataFrame()
        obs['Cell_ID'] = CellID
        var = pd.DataFrame(index = GeneName)
        self.scRNAseq_Counts = ad.AnnData(np.array(scRNAseqTmp).T, obs = obs, var = var)
        fh.close()

    def Intersection(self, lst1, lst2):
        return set(lst1).intersection(lst2)

    def diff(self, first, second):
        second = set(second)
        return [item for item in first if item not in second]

    def ReadTurth(self, FileName, CellColInd, ClusterColInd, lineskip=0):
        fh = open(FileName, 'r')
        for i in range(lineskip):
            fh.readline()

        CellLabel = {}
        CellIDAll = list(self.scRNAseq_Counts.obs['Cell_ID'])
        for tline in fh:
            line = tline.strip().split()
            if line[CellColInd] in CellIDAll:
                CellLabel[line[CellColInd]] = line[ClusterColInd]
                #print(line[CellColInd])

        ClusterLabel = list()
        NumNone = 0
        for cellid in self.scRNAseq_Counts.obs['Cell_ID']:
            if cellid in CellLabel:
                ClusterLabel.append(CellLabel[cellid])
            else:
                ClusterLabel.append('None')
                NumNone = NumNone + 1
        if NumNone > 0:
            print("None number is %d " % NumNone)

        self.scRNAseq_Counts.obs['Cell_Label'] = ClusterLabel

        self.ClusterLabel = list(self.scRNAseq_Counts.obs['Cell_Label'])

        fh.close()

    def DefineSuperCell_HierarchyClustering_HVG(self, NumSuperCell=100):
        XCellSingle = self.scRNAseq_Propcessed.X
        XCellSingleNorm = np.log1p(10000* ((XCellSingle) / XCellSingle.sum(axis=1)[:,np.newaxis]))
        Z = linkage(XCellSingleNorm, 'ward')

        self.SuperClusterinOne = fcluster(Z, NumSuperCell, criterion='maxclust')
        unique, counts = np.unique(self.SuperClusterinOne, return_counts=True)
        #print(dict(zip(unique, counts)))

        Num_HVG_Genes = self.scRNAseq_HVGData_log1p.X.shape[1]
        scRNAseqSuper = np.zeros((NumSuperCell, Num_HVG_Genes))
        self.NumSuperCell = NumSuperCell
        SCid = list()
        self.SuperCluster = {}
        for i in range(NumSuperCell):
            cid = i + 1
            self.SuperCluster[i] = np.nonzero(self.SuperClusterinOne==cid)[0]
            ExpG = self.scRNAseq_HVGData_log1p.X[self.SuperCluster[i],]
            scRNAseqSuper[i,:] = np.mean(ExpG, axis = 0)
            SCid.extend([i])
        obs = pd.DataFrame()
        obs['SuperCellID'] = SCid
        self.scRNAseq_SuperHVGLog1P = ad.AnnData(scRNAseqSuper, obs = obs)
        return self.SuperClusterinOne

    def DefineSuperCell_HierarchyClustering_HVGLog1PPCA(self, NumSuperCell=100):
        XCellSingleNorm = pca(self.scRNAseq_HVGData_log1p.X, n_comps=50)
        print(XCellSingleNorm.shape)
        Z = linkage(XCellSingleNorm, 'ward')

        self.SuperClusterinOne = fcluster(Z, NumSuperCell, criterion='maxclust')
        unique, counts = np.unique(self.SuperClusterinOne, return_counts=True)
        #print(dict(zip(unique, counts)))

        Num_HVG_Genes = self.scRNAseq_HVGData_log1p.X.shape[1]
        scRNAseqSuper = np.zeros((NumSuperCell, 50))
        scRNAseqSuper_noPCA = np.zeros((NumSuperCell, len(self.scRNAseq_HVGData_log1p.var)))
        scRNAseqSuper_RawCount = np.zeros((NumSuperCell, len(self.scRNAseq_Counts.var)))
        self.NumSuperCell = NumSuperCell
        SCid = list()
        self.SuperCluster = {}
        for i in range(NumSuperCell):
            cid = i + 1
            self.SuperCluster[i] = np.nonzero(self.SuperClusterinOne==cid)[0]
            ExpG = XCellSingleNorm[self.SuperCluster[i],]
            ExpG_nopca = self.scRNAseq_HVGData_log1p.X[self.SuperCluster[i],]
            RawCount = self.scRNAseq_Counts.X[self.SuperCluster[i],]
            scRNAseqSuper[i,:] = np.mean(ExpG, axis = 0)
            scRNAseqSuper_noPCA[i,:] = np.mean(ExpG_nopca, axis = 0)
            #scRNAseqSuper_RawCount[i,:] = np.mean(RawCount, axis = 0)
            SCid.extend([i])
        obs = pd.DataFrame()
        obs['SuperCellID'] = SCid
        self.scRNAseq_SuperHVGLog1P = ad.AnnData(scRNAseqSuper, obs = obs)
        self.scRNAseq_SuperHVGLog1P_noPca = ad.AnnData(scRNAseqSuper_noPCA, obs = obs, var = self.scRNAseq_HVGData_log1p.var)
        #self.scRNAseq_SuperRawCount = ad.AnnData(scRNAseqSuper_RawCount, obs = obs, var = self.scRNAseq_Counts.var)
        return self.SuperClusterinOne

    def SuperCell_TurthLabel(self):
        NumSC = len(self.SuperCluster)
        SuperClusterLabel = list()
        for i in range(NumSC):
            Tmp = self.scRNAseq_Counts.obs['Cell_Label'][self.SuperCluster[i]]
            unique, counts = np.unique(Tmp, return_counts=True)
            Dic = dict(zip(unique, counts))
            labelt = sorted(Dic.items(), key=operator.itemgetter(1))[-1][0]
            SuperClusterLabel.extend([labelt])

        Uni_SCL = list(set(SuperClusterLabel))
        Map = {}
        for i in range(len(Uni_SCL)):
            Map[Uni_SCL[i]] = i

        self.SuperClusterLabelNum = np.zeros(len(SuperClusterLabel))
        for i in range(len(SuperClusterLabel)):
            self.SuperClusterLabelNum[i] = int(Map[SuperClusterLabel[i]])

        self.SuperCell_AssignMat = np.zeros((len(SuperClusterLabel), len(Uni_SCL)))
        for i in range(len(Uni_SCL)):
            ind = np.where(self.SuperClusterLabelNum==i)[0]
            self.SuperCell_AssignMat[ind,i] = 1.

    def Align2CoExpGene_SuperCellLevel(self, CoExpGene):
        Idt = list()
        HVG = list(self.scRNAseq_SuperHVGLog1P_noPca.var_names)
        for cogene in CoExpGene:
            Idt.extend([HVG.index(cogene)])
        self.scRNAseqSuperSelectAligned = self.scRNAseq_SuperHVGLog1P_noPca.X[:, Idt]

    def Construct_SuperAdjMat_HVGLog1PPCA(self):
        Dis = euclidean_distances(self.scRNAseq_SuperHVGLog1P.X)#np.corrcoef(self.scRNAseq_SuperHVGLog1P.X)#cosine_similarity(self.scRNAseq_SuperHVGLog1P.X)
        self.SuperAdjMat_HVGLog1PPCA = 1. / (1. + Dis)
        return self.SuperAdjMat_HVGLog1PPCA

    def Construct_SuperDisMat_HVGLog1PPCA(self):
        self.SuperDisMat_HVGLog1PPCA = cosine_distances(self.scRNAseq_SuperHVGLog1P.X)#np.corrcoef(self.scRNAseq_SuperHVGLog1P.X)#cosine_similarity(self.scRNAseq_SuperHVGLog1P.X)
        return self.SuperDisMat_HVGLog1PPCA

    def SNN_HVGLog1PPCA(self, k):
        self.SNN_HVGLog1PPCA = shared_nearest_neighbors(self.SuperAdjMat_HVGLog1PPCA, k, metric='similarity')
        return self.SNN_HVGLog1PPCA

    def ModularityMat(self, AdjMat, Lambda):
        #np.fill_diagonal(AdjMat, 0)
        Dv = np.sum(AdjMat, axis = 0)[:, np.newaxis]
        Wsum = np.sum(AdjMat)
        PMat = Dv.dot(Dv.T) / Wsum
        self.MMat = AdjMat - Lambda*PMat
        return self.MMat

    def Build_Dense_Separated_SuperCell_AdjMat(self):
        DisMat = self.Construct_SuperDisMat_HVGLog1PPCA()
        SimMat = 1-DisMat
        SimMat[SimMat<0.2] = 0.
        np.fill_diagonal(SimMat, 1.)
        PRTest = PageRankVec(SimMat, 0.8, 1e-5)
        LocalDen = np.zeros(self.NumSuperCell)
        PRVecDen = np.zeros((self.NumSuperCell,self.NumSuperCell))
        for i in range(self.NumSuperCell):
            #print("node %d" % (i))
            PRVec = PRTest.ApproximatePR_Vertex(i)
            PRVecDen[:,i], LocalDen[i] = PRTest.Sweep_Density(PRVec, np.count_nonzero(PRVec))
        self.DenMat = np.zeros((self.NumSuperCell,self.NumSuperCell))
        for i in range(self.NumSuperCell):
            self.DenMat = self.DenMat + np.outer(PRVecDen[:,i], PRVecDen[:,i])
        return self.DenMat

    def SDP_ADMM_Modularity(self, MMat):
        self.MResult = ADMM_Modulaity_SDP(MMat)
        return self.MResult

    def Rounding_Modularity(self, NumClust):
        optimalobj = -1E10
        self.NumClust = NumClust
        for i in range(100):
            kmeans = KMeans(n_clusters=NumClust).fit(self.MResult)#REX#Xc+Yc
            SingleCLabel = kmeans.labels_

            AssignMat = np.zeros((SingleCLabel.shape[0],NumClust))
            for i in range(NumClust):
                Ind = np.where(SingleCLabel==i)
                AssignMat[Ind[0],i] = 1.

            Obj = np.trace(AssignMat.T.dot(self.MMat).dot(AssignMat))
            print(Obj)
            if Obj > optimalobj:
                optimalobj = Obj
                self.ModularityResult = SingleCLabel
        print(optimalobj)
        return self.ModularityResult

    def Cluster_AssignMat(self, NumClust):
        SingleCLabel = self.ModularityResult
        AssignMat = np.zeros((SingleCLabel.shape[0],NumClust))
        for i in range(NumClust):
            Ind = np.where(SingleCLabel==i)
            AssignMat[Ind[0],i] = 1.

        return AssignMat

    def SuperCellLabe_CellLevelLabel(self, SuperCellLabel):
        SCLabel = np.array(SuperCellLabel)
        SCLabel_Uids = np.unique(SCLabel)
        self.CLLabel = np.zeros(self.NumCell)
        for i in range(SCLabel_Uids.size):
            SCIdList = np.where(SCLabel == i)[0]
            TmpList = list()
            for j in SCIdList:
                TmpList.extend(self.SuperCluster[j])
            self.CLLabel[TmpList] = i
        return self.CLLabel

    def ARIScore(self):
        ClusterLabel = list(self.scRNAseq_Counts.obs['Cell_Label'])
        UniLabel = list(set(ClusterLabel))
        Mapping = {}
        for i in range(len(UniLabel)):
            Mapping[UniLabel[i]] = i

        Turth = list()
        for Cid in ClusterLabel:
            Turth.extend([Mapping[Cid]])

        Turth = np.array(Turth)
        ARI = adjusted_rand_score(self.CLLabel, Turth)
        return ARI

    def UndersegError(self, SuperCellClustering):
        NumCell = self.NumCell
        self.ClusterLabel = list(self.scRNAseq_Counts.obs['Cell_Label'])

        #SuperCell
        USCellid = np.unique(SuperCellClustering)
        SCClustering = {}
        for i in USCellid:
            tmpid = np.where(SuperCellClustering==i)[0]
            SCClustering[i] = tmpid
        #print(SCClustering)

        #Turth
        UniLabel = list(set(self.ClusterLabel))
        Mapping = {}
        for i in range(len(UniLabel)):
            Mapping[UniLabel[i]] = i
        Turth = list()
        for Cid in self.ClusterLabel:
            Turth.extend([Mapping[Cid]])
        Turth = np.array(Turth)
        UniTurth = np.unique(Turth)
        TurthClustering = {}
        for i in UniTurth:
            tmpid = np.where(Turth==i)[0]
            TurthClustering[i] = tmpid
        #print(TurthClustering)

        AllE = 0.
        for Tid in TurthClustering:
            for Cid in SCClustering:
                InterSect = np.intersect1d(TurthClustering[Tid], SCClustering[Cid])
                NumIS = len(InterSect)
                NumDif = len(SCClustering[Cid]) - NumIS
                #print("Tid %d, Cid %d, InterS %d, Diff %d" %(Tid, Cid, NumIS, NumDif))
                if NumIS != 0:
                    AllE = AllE + min(NumIS, NumDif)
        Error = AllE / NumCell

        return Error


    def ASA(self, SuperCellClustering):
        NumCell = self.NumCell
        self.ClusterLabel = list(self.scRNAseq_Counts.obs['Cell_Label'])
        #SuperCell
        USCellid = np.unique(SuperCellClustering)
        SCClustering = {}
        for i in USCellid:
            tmpid = np.where(SuperCellClustering==i)[0]
            SCClustering[i] = tmpid
        #print(SCClustering)

        #Turth
        UniLabel = list(set(self.ClusterLabel))
        Mapping = {}
        for i in range(len(UniLabel)):
            Mapping[UniLabel[i]] = i
        Turth = list()
        for Cid in self.ClusterLabel:
            Turth.extend([Mapping[Cid]])
        Turth = np.array(Turth)
        UniTurth = np.unique(Turth)
        TurthClustering = {}
        for i in UniTurth:
            tmpid = np.where(Turth==i)[0]
            TurthClustering[i] = tmpid
        #print(TurthClustering)

        # compute
        ALLT = 0.
        for Cid in SCClustering:
            MaxT = 0.
            for Tid in TurthClustering:
                InterSect = np.intersect1d(TurthClustering[Tid], SCClustering[Cid])
                NumIS = len(InterSect)
                if NumIS > MaxT:
                    MaxT = NumIS
            ALLT = ALLT + MaxT

        Acc = ALLT / NumCell
        return Acc

class MergeSingleCell:
    def __init__(self, *args):
        self.MSinglCell = args
        self.NumDataSets = len(self.MSinglCell)

    def GroundTruthMSingleCell(self):
        ClusterLabelAcross = list()
        for singlecell in self.MSinglCell:
            ClusterLabelAcross = list(set(ClusterLabelAcross)|set(singlecell.ClusterLabel))
        #print(ClusterLabelAcross)
        MapAcross={}
        for i in range(len(ClusterLabelAcross)):
            MapAcross[ClusterLabelAcross[i]] = i

        TruthCluster = list()
        for singlecell in self.MSinglCell:
            for cid in singlecell.ClusterLabel:
                TruthCluster.extend([MapAcross[cid]])
        return np.asarray(TruthCluster)

    def MultiDefineSuperCell(self, *nsc):
        Ind = 0
        for singlecell in self.MSinglCell:
            singlecell.DefineSuperCell_HierarchyClustering_HVGLog1PPCA(NumSuperCell=nsc[Ind])
            Ind = Ind + 1

    def ConstructBetweenSimiarlityMat_SuperCellLevel(self, solution=1.0):
        NumSCell = [0]
        for singlecell in self.MSinglCell:
            NumSCell.extend([singlecell.NumSuperCell])
        NumSCell = np.array(NumSCell)
        AccNumSCell = np.cumsum(NumSCell)
        self.AccNumSCell = AccNumSCell

        NumTotalSCell = AccNumSCell[-1]
        self.NumTotalSCell = NumTotalSCell
        self.BMat = np.zeros((NumTotalSCell, NumTotalSCell))
        #self.AMod = np.zeros((NumTotalSCell, NumTotalSCell))
        #self.ModularityMat_Between_SuperCell = np.zeros((NumTotalSCell, NumTotalSCell))

        NumDataset = len(self.MSinglCell)
        RList = np.arange(0,NumDataset-1)
        for rid in RList:
            CList = np.arange(rid+1,NumDataset)
            for cid in CList:
                if(rid == cid):
                    continue
                # align genes for pairwise datasets
                HVG1 = list(self.MSinglCell[rid].scRNAseq_SuperHVGLog1P_noPca.var_names)
                HVG2 = list(self.MSinglCell[cid].scRNAseq_SuperHVGLog1P_noPca.var_names)
                CoExpGenes = list(set(HVG1).intersection(HVG2))
                self.MSinglCell[rid].Align2CoExpGene_SuperCellLevel(CoExpGenes)
                self.MSinglCell[cid].Align2CoExpGene_SuperCellLevel(CoExpGenes)

                # normlize
                XSuperAcross = (copy.deepcopy(self.MSinglCell[rid].scRNAseqSuperSelectAligned))
                YSuperAcross = (copy.deepcopy(self.MSinglCell[cid].scRNAseqSuperSelectAligned))
                XYStack = np.vstack((XSuperAcross,YSuperAcross))
                XYStack_PCA = pca(XYStack, n_comps=30)

                # pairwise similarity matrix
                K = cosine_similarity(XYStack_PCA[0:self.MSinglCell[rid].NumSuperCell,:], XYStack_PCA[self.MSinglCell[rid].NumSuperCell:,:])
                K[K<0] = 0.
                self.BMat[AccNumSCell[rid]:AccNumSCell[rid+1],AccNumSCell[cid]:AccNumSCell[cid+1]] = K
                # Across Modulairty
                #AvgWeight = np.sum(K) / (NumSCell[rid+1]*NumSCell[cid+1])
                #self.AMod[AccNumSCell[rid]:AccNumSCell[rid+1],AccNumSCell[cid]:AccNumSCell[cid+1]] = K - np.ones(K.shape)*AvgWeight
                #DvecRow = np.sum(K, axis=1)[:,np.newaxis]
                #DvecCol = np.sum(K, axis=0)[:,np.newaxis]
                #PMat = (1.0*solution/np.sum(K)) * DvecRow.dot(DvecCol.T)
                #self.ModularityMat_Between_SuperCell[AccNumSCell[rid]:AccNumSCell[rid+1],AccNumSCell[cid]:AccNumSCell[cid+1]] = K - PMat
        #self.ModularityMat_Between_SuperCell = self.ModularityMat_Between_SuperCell + self.ModularityMat_Between_SuperCell.T
        self.BMat_Between_SuperCell = self.BMat + self.BMat.T
        return self.BMat_Between_SuperCell


    def ConstructWithinSimiarlityMat_SuperCellLevel(self, solution=1.0):
        DenMat = self.MSinglCell[0].Build_Dense_Separated_SuperCell_AdjMat()
        self.WSCN_Within_SuperCell = DenMat#self.MSinglCell[0].ModularityMat(DenMat, solution)
        NumDataset = len(self.MSinglCell)
        for i in range(1, NumDataset):
            DenMatT = self.MSinglCell[i].Build_Dense_Separated_SuperCell_AdjMat()
            #DenMatTT = shared_nearest_neighbors(DenMatT)
            #MDenMatT = self.MSinglCell[i].ModularityMat(DenMatT, solution)
            #self.ModularityMat_Within_SuperCell = block_diag(self.ModularityMat_Within_SuperCell, MDenMatT)
            self.WSCN_Within_SuperCell = block_diag(self.WSCN_Within_SuperCell, DenMatT)

        return self.WSCN_Within_SuperCell

    def SDP_NKcut(self, K, Lambda=1.0):
        BigWithin = self.WSCN_Within_SuperCell
        DWithinraw = np.sum(BigWithin, axis = 0)
        DWithinraw[DWithinraw==0] = 1.
        DWithin = np.diag(DWithinraw**(-0.5))
        DWithindiag = np.sum(BigWithin, axis = 0)**(0.5)
        I = np.eye(self.NumTotalSCell)
        self.LWithinbar = I - DWithin.dot(BigWithin).dot(DWithin)

        BigBetween = self.BMat_Between_SuperCell
        BigBetween[BigBetween<0] = 0.
        DBetweenraw = np.sum(BigBetween, axis = 0)
        DBetweenraw[DBetweenraw==0] = 1.
        DBetween = np.diag(DBetweenraw**(-0.5))
        DBetweendiag = np.sum(BigBetween, axis = 0)**(0.5)
        self.LBetweenbar = I - DBetween.dot(BigBetween).dot(DBetween)

        self.SDPKcutResult = SDPMerge(self.LWithinbar, self.LBetweenbar, DWithindiag, DBetweendiag, K, Lambda)

    def NKcut_Rounding(self, Lambda, NumCluster):
        optimalobj = 1E10
        score = -100
        optimalden = -1E10
        self.NComClust = NumCluster
        for i in range(100):
            print("#########")
            kmeans = KMeans(n_clusters=NumCluster).fit(self.SDPKcutResult)#REX#Xc+Yc
            SingleCLabel = kmeans.labels_

            AssignMat = np.zeros((self.NumTotalSCell,NumCluster))
            for i in range(NumCluster):
                Ind = np.where(SingleCLabel==i)
                AssignMat[Ind[0],i] = 1.

            DenBetweenMat = AssignMat.T.dot(self.BMat_Between_SuperCell).dot(AssignMat)
            CDenBetween = np.diag(DenBetweenMat)
            Cnum = np.sum(AssignMat, axis=0)
            DenBetween = CDenBetween / Cnum[:,np.newaxis]
            AvgDenBetween = np.mean(DenBetween)
            DenWithinMat = AssignMat.T.dot(self.WSCN_Within_SuperCell).dot(AssignMat)
            CDenWithin = np.diag(DenWithinMat)
            Cnum = np.sum(AssignMat, axis=0)
            DenWithin = CDenWithin / Cnum[:,np.newaxis]
            AvgDenWithin = np.mean(DenWithin)
            if AvgDenBetween*AvgDenWithin > optimalden:
                optimalden = AvgDenBetween*AvgDenWithin
                optdenRES = SingleCLabel

            Obj = np.trace(AssignMat.T.dot(self.LBetweenbar + Lambda*self.LWithinbar).dot(AssignMat)) #+ Lambda*LCbar
            print(Obj)
            if Obj < optimalobj:
                optimalobj = Obj
                OptiamlRes = SingleCLabel

            self.ClusterResult = OptiamlRes#optdenRES #################need to check
        #print(optimalobj)
        return OptiamlRes, optdenRES

    def PostProcessing(self, RoundingClustering):# multiple datasets
        DisMat = self.MSinglCell[0].Construct_SuperDisMat_HVGLog1PPCA()
        SimMat = 1-DisMat
        SimMat[SimMat<0] = 0
        SimBig = SimMat
        for i in range(1, self.NumDataSets):
            DisMat = self.MSinglCell[0].Construct_SuperDisMat_HVGLog1PPCA()
            SimMat = 1-DisMat
            SimMat[SimMat<0] = 0
            SimBig = block_diag(SimBig, SimMat)
        BigWithin = self.WSCN_Within_SuperCell
        BigCross = self.BMat_Between_SuperCell
        DataSetIndx = self.AccNumSCell
        UCid = np.unique(RoundingClustering)
        NumUClust = len(UCid)
        NumDataset = len(DataSetIndx) - 1

        CidperData = [dict() for i in range(NumDataset)]
        for i in range(NumDataset):
            CidperData[i] = RoundingClustering[DataSetIndx[i]: DataSetIndx[i+1]]

        NewRoundingClustering = copy.deepcopy(RoundingClustering)
        for i in range(NumDataset):
            print("###########Dataset %d##############"%(i))
            for j in range(NumUClust):
                print("************Cluster %d************"%(j))
                idwithin = np.where(CidperData[i] == UCid[j])[0]
                idoverall = idwithin+DataSetIndx[i]
                if(len(idwithin) == 0):
                    print("here!!!")
                    continue

                # within similary
                idoverallfetch = np.ix_(idoverall, idoverall)
                WithinMat = BigWithin[idoverallfetch]
                NumComp, CompId = connected_components(WithinMat)
                #print("%d connected compoenent!!!"%(NumComp))
                if(NumComp <= 1):
                    print("Only one Compoenent!!!")
                    continue
                print("%d connected compoenent!!!"%(NumComp))
                UCompId = np.unique(CompId)
                CompOverallId = [dict() for ii in range(len(UCompId))]
                for ii in range(len(UCompId)):
                    localid = np.where(CompId==ii)[0]
                    tmpid = np.ix_(localid)
                    CompOverallId[ii] = idoverall[tmpid]

                # check each compoent
                for k in range(len(UCompId)):
                    CompIdOverall = CompOverallId[k]
                    print(CompIdOverall)
                    Score = 0
                    BestCid = -1
                    for l in range(NumUClust):
                        Cidwithinlocal = np.where(CidperData[i]==l)[0]
                        if len(Cidwithinlocal) ==0:
                            continue
                        Cidwithinoverall = Cidwithinlocal+DataSetIndx[i]
                        Cidoverall = np.where(RoundingClustering==l)[0]

                        # Within
                        idtmp1 = np.ix_(CompIdOverall, Cidwithinoverall)
                        MatWithin = SimBig[idtmp1]
                        MatWithinTmp = SimBig[CompIdOverall,:]
                        x1, y1 = MatWithin.shape
                        Stmp1 = np.sum(MatWithin) / np.sum(MatWithinTmp)#(x1*y1)

                        # between
                        idtmp2 = np.ix_(CompIdOverall, Cidoverall)
                        MatBetween = BigCross[idtmp2]
                        MatBetweenTmp = BigCross[CompIdOverall,:]
                        x2, y2 = MatBetween.shape
                        Stmp2 = np.sum(MatBetween) / np.sum(MatBetweenTmp)#(x2*y2)

                        print(l, Stmp1, Stmp2)

                        if(Score < Stmp1 + Stmp2 ):#
                            Score = Stmp1+ Stmp2#
                            BestCid = l
                    print("Best")
                    print(Score, BestCid)
                    NewRoundingClustering[np.ix_(CompIdOverall)] = BestCid
        return NewRoundingClustering

    def FindCommonCluster_Modularity(self, labmda = 2.0):
        PMatAll = self.ModularityMat_Between_SuperCell + labmda*self.ModularityMat_Within_SuperCell
        self.PMat = PMatAll
        self.SDPResult = ADMM_Modulaity_SDP(PMatAll)

    def RoundingSDP_Modularity(self, NComClust):
        optimalobj = 0.
        self.NComClust = NComClust
        for i in range(100):
            kmeans = KMeans(n_clusters=NComClust).fit(self.SDPResult)#REX#Xc+Yc
            MergeCLabel = kmeans.labels_

            AssignMat = np.zeros((MergeCLabel.shape[0],NComClust))
            for i in range(NComClust):
                Ind = np.where(MergeCLabel==i)
                AssignMat[Ind[0],i] = 1.

            Obj = np.trace(AssignMat.T.dot(self.PMat).dot(AssignMat))
            if Obj > optimalobj:
                optimalobj = Obj
                self.ClusterResult = MergeCLabel
        return self.ClusterResult

    def Evaluation(self, CResult):
        self.ClusterResult = CResult
        NumDataset = len(self.MSinglCell)
        MatchedCell_Data = {}
        for j in range(NumDataset):
            MatchedCell_Data[j] = 0.

        NumCDataj = np.zeros(NumDataset)
        NumComC = 0.
        AvgAccComC = 0.
        AvgAccj = np.zeros(NumDataset)
        for i in range(self.NComClust):
            CellLabel_Dataj = {}
            CellLabel_All = list()

            print("##################")
            sorted_cj = [dict() for x in range(NumDataset)]
            for j in range(NumDataset):
                print("Dataset %d"%j)
                TLabel_Dataj = list(self.MSinglCell[j].scRNAseq_Counts.obs['Cell_Label'])
                SCellLabel_Clusteri_Datasetj = np.where(self.ClusterResult[self.AccNumSCell[j]:self.AccNumSCell[j+1]]==i)[0]
                Tmp = list()
                for k in SCellLabel_Clusteri_Datasetj:
                        Tmp.extend(self.MSinglCell[j].SuperCluster[k])
                Tmp_lab = [TLabel_Dataj[n] for n in Tmp]
                NumTmp = len(Tmp_lab)
                if NumTmp > 10:
                    CellLabel_Dataj =collections.Counter(Tmp_lab)
                    for keyj in CellLabel_Dataj:
                        CellLabel_Dataj[keyj] = CellLabel_Dataj[keyj] / NumTmp
                    sorted_cj[j] = sorted(CellLabel_Dataj.items(), key=operator.itemgetter(1))
                    print(NumTmp)
                    print(sorted_cj[j])
                    AvgAccj[j] = AvgAccj[j] + sorted_cj[j][-1][1]
                    NumCDataj[j] = NumCDataj[j] + 1.

            Ind = 0
            for j in range(NumDataset):
                if(len(sorted_cj[j]) == 0):
                    Ind = 1
            if Ind == 1:
                continue
            TlabelC = list()
            AvgSumT = 0.
            for j in range(NumDataset):
                TlabelC.append(sorted_cj[j][-1][0])
                AvgSumT = AvgSumT + sorted_cj[j][-1][1]
            if len(list(set(TlabelC))) == 1:
                NumComC = NumComC + 1.
            AvgAccComC = AvgAccComC + AvgSumT/NumDataset

        for j in range(NumDataset):
            print("Dataset %i: Accuracy %f" % (j, AvgAccj[j]/NumCDataj[j]))
        print("Common Cluster Accuracy %f" % (AvgAccComC/NumComC))
