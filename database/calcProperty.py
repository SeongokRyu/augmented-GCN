import numpy as np
from rdkit import Chem
from rdkit.Chem.QED import qed
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Contrib.SA_Score.sascorer import calculateScore
import sys


smiles_f = open('./ZINC/smiles.txt')
smiles_list = smiles_f.readlines()

logPList = []
molWtList = []
TPSAList = []
QEDList = []
SASList = []
for smi in smiles_list:
    smi = smi.strip()
    m = Chem.MolFromSmiles(smi)
    molWt = ExactMolWt(m)
    logP = MolLogP(m)
    TPSA = CalcTPSA(m)
    _qed = qed(m)
    sas = calculateScore(m)

    logPList.append(logP)
    molWtList.append(molWt)
    TPSAList.append(TPSA)
    QEDList.append(_qed)
    SASList.append(sas)

logPList = np.asarray(logPList)
TPSAList = np.asarray(TPSAList)
QEDList = np.asarray(QEDList)
SASList = np.asarray(SASList)

np.save('./ZINC/logP.npy', logPList)
np.save('./ZINC/TPSA.npy', TPSAList)
np.save('./ZINC/QED.npy', QEDList)
np.save('./ZINC/SAS.npy', SASList)
