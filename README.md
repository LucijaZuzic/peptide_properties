# Peptide self assembly prediction
Cilj projekta je razvoj modela strojnog učenja koji uspješno predviđa sposobnost self assemblyja peptidnih sekvenca.

## Priprema radnog okruženja
Nakon stvaranja conda virtualnog okruženja, potrebno je instalirati sljedeće knjižnice funkcija:
```console
conda install matplotlib==3.4.2  
conda install keras==2.4.3  
conda install -c conda-forge scikit-learn==0.24.2  
  
# Newer versions of NumPy create a Tensorflow error.  
# https://github.com/tensorflow/models/issues/9706  
conda install -c conda-forge numpy==1.19.5  
  
pip3 install pydot==1.4.1  
  
# Install graphviz system-wide. Graphviz bin folder should be in PATH.  
sudo apt install graphviz
```

## Organizacija projekta
- **`code`** &#8594; sadrži Python kod projekta, glavna skripta je `model_train_and_test.py`.
- **`data`** &#8594; sadrži podatke. Konkretno, sadržani su aggregation propensity scores aminokiselina, dipeptida i tripeptida, te sekvenece peptida uz pripadajuću oznaku (1 - ima self assembly, 0 - nema self-assembly)
- **`model_data`** &#8594; navedena mapa služi za pohranu podataka. Konkretno, sprema se najbolji model prilikom treninga i vizualni prikaz modela generiran naredbom `keras.utils.plot_model`.

---

## Datoteka environment.yml

Uz korištenje anaconde i datoteke environment.yml (prilagođena za Windows), moguće je stvoriti novo radno okruženje s istim instaliranim paketima koji su navedeni u datoteci.

To se izvodi u poddirektoriju code naredbom:

```
conda env create --name <envname> --file environment.yml
```