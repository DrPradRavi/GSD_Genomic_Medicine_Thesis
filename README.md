# GSD_Genomic_Medicine_Thesis
This repository contains the scripts used in my MSt in Genomic Medicine Thesis. 
1. To gather the relevant samples from the Genomics England Research Environment and pre-process them, kindly run 'run.py' within pre_processing folder. This would run each file within that folder in order.
2. Feature selection is done using random forest and tuned RelieFf algorithms.
3. Machine learning to select the final important features is done using random forest and XGBoost.
4. To identify epistasis groups, run MI_ord.py within the Mutual information folder.
5. For bio-analysis of the results using ENSEMBL, download a GRCh38 GTF file and run 'Condensing_GTF_file.py' before running ENSEMBL.py.
6. Lastly to compare the results to a traditional GWAS, run the two scripts which runs a CHI2 and Fisher's GWAS.
