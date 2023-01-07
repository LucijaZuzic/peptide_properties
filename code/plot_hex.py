import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import PATH_TO_EXTENSION, DATA_PATH, SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH, setSeed, predictions_name
from custom_plots import merge_type_iteration 
from scipy import stats
import sklearn

def hex_predictions_name(some_path, number, final_model_type, iteration):
    return predictions_name(some_path, number, final_model_type, iteration).replace("predictions", "hex_predictions")
 
def hex_predictions_png_name(some_path, number, final_model_type, iteration):
    return hex_predictions_name(some_path, number, final_model_type, iteration).replace(".txt", ".png")

def hex_predictions_final_name(some_path):
    return "../seeds/all_seeds/" + PATH_TO_EXTENSION[some_path] + "_all_tests_hex_predictions.png"

def hex_predictions_seed_name(some_path):
    return "../seeds/seed_" + str(seed) + some_path.replace("..", "") + PATH_TO_EXTENSION[some_path] + "_seed_" + str(seed) + "_hex_predictions.png"

def myfunc(x):
  return slope * x + intercept

df = pd.read_csv(DATA_PATH + "41557_2022_1055_MOESM3_ESM_Figure3a_5mer_score_shortMD.csv")

dict_hex = {}
for i in df['pep']:
    dict_hex[i] = '1' 

seq_example = ''
for i in range(24):
    seq_example += 'A'
dict_hex[seq_example] = '1' 

best_batch_size = 600
best_model = '' 
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276] 
NUM_TESTS = 5

offset = 1 
  
properties = np.ones(95)
masking_value = 2

actual_AP = []

for i in df['AP']:
    actual_AP.append(i) 

actual_AP_long = []

for seed in seed_list:
    for number in range(1, NUM_TESTS + 1):
        for i in df['AP']:
            actual_AP_long.append(i) 

actual_AP_mid = []
 
for number in range(1, NUM_TESTS + 1):
    for i in df['AP']:
        actual_AP_mid.append(i) 

model_predictions_seq_hex = []
model_predictions_AP_hex = []
model_predictions_hex = []

model_predictions_seq_hex_seed = {}
model_predictions_AP_hex_seed = {}
model_predictions_hex_seed = {}

for seed in seed_list:
    setSeed(seed)        
    print("Seed " + str(seed))

    model_predictions_seq_hex_seed[seed] = []
    model_predictions_AP_hex_seed[seed] = []
    model_predictions_hex_seed[seed] = []

    for number in range(1, NUM_TESTS + 1):
        print("Seed " + str(seed) + " Test " + str(number))

        model_predictions_seq_hex_file = open(hex_predictions_name(SEQ_MODEL_DATA_PATH, number, 'weak', 1), 'r')
        model_predictions_AP_hex_file = open(hex_predictions_name(MY_MODEL_DATA_PATH, number, 'weak', 1), 'r')
        model_predictions_hex_file = open(hex_predictions_name(MODEL_DATA_PATH, number, 'weak', 1), 'r')

        model_predictions_seq_hex_lines = model_predictions_seq_hex_file.readlines()
        model_predictions_AP_hex_lines = model_predictions_AP_hex_file.readlines()
        model_predictions_hex_lines = model_predictions_hex_file.readlines()

        model_predictions_seq_hex_one = eval(model_predictions_seq_hex_lines[0])[:-1]
        model_predictions_AP_hex_one = eval(model_predictions_AP_hex_lines[0])[:-1]
        model_predictions_hex_one = eval(model_predictions_hex_lines[0])[:-1]

        for x in model_predictions_seq_hex_one:
            model_predictions_seq_hex.append(x)
            model_predictions_seq_hex_seed[seed].append(x)

        for x in model_predictions_AP_hex_one:
            model_predictions_AP_hex.append(x)
            model_predictions_AP_hex_seed[seed].append(x)

        for x in model_predictions_hex_one:
            model_predictions_hex.append(x)
            model_predictions_hex_seed[seed].append(x)

        plt.figure()
        plt.title(
            merge_type_iteration(SEQ_MODEL_DATA_PATH, 'weak', 1, number).replace(" Weak 1", "")  
            + "\nPredicted self assembly probability for hexapeptides"
        )
        plt.xlabel("Predicted self assembly probability")
        plt.ylabel("AP")
        plt.scatter(
            model_predictions_seq_hex_one,
            actual_AP
        ) 
        slope, intercept, r, p, std_err = stats.linregress(model_predictions_seq_hex_one, actual_AP)
        print("Seq. model R: " + str(r))
        print("Seq. model corrcoef: " + str(np.corrcoef(model_predictions_seq_hex_one, actual_AP)[0][1])) 
        print("Seq. model spearmanr: " + str(stats.spearmanr(model_predictions_seq_hex_one, actual_AP)[0])) 
        print("Seq. model R2: " + str(sklearn.metrics.r2_score(model_predictions_seq_hex_one, actual_AP)))
        mymodel = list(map(myfunc, model_predictions_seq_hex_one)) 
        plt.plot(
            model_predictions_seq_hex_one,
            mymodel, color = 'r'
        )  
        plt.savefig(hex_predictions_png_name(SEQ_MODEL_DATA_PATH, number, 'weak', 1))
        plt.close()

        plt.figure()
        plt.title(
            merge_type_iteration(MODEL_DATA_PATH, 'weak', 1, number).replace(" Weak 1", "")  
            + "\nPredicted self assembly probability for hexapeptides"
        )
        plt.xlabel("Predicted self assembly probability")
        plt.ylabel("AP")
        plt.scatter(
            model_predictions_hex_one,
            actual_AP
        ) 
        slope, intercept, r, p, std_err = stats.linregress(model_predictions_hex_one, actual_AP)
        print("Seq. and AP model R: " + str(r)) 
        print("Seq. and AP model corrcoef: " + str(np.corrcoef(model_predictions_hex_one, actual_AP)[0][1])) 
        print("Seq. and AP model spearmanr: " + str(stats.spearmanr(model_predictions_hex_one, actual_AP)[0])) 
        print("Seq. and AP model R2: " + str(sklearn.metrics.r2_score(model_predictions_hex_one, actual_AP)))  
        mymodel = list(map(myfunc, model_predictions_hex_one)) 
        plt.plot(
            model_predictions_hex_one,
            mymodel, color = 'r'
        )  
        plt.savefig(hex_predictions_png_name(MODEL_DATA_PATH, number, 'weak', 1))
        plt.close() 

        plt.figure()
        plt.title(
            merge_type_iteration(MY_MODEL_DATA_PATH, 'weak', 1, number).replace(" Weak 1", "")   
            + "\nPredicted self assembly probability for hexapeptides"
        )
        plt.xlabel("Predicted self assembly probability")
        plt.ylabel("AP")
        plt.scatter(
            model_predictions_AP_hex_one,
            actual_AP
        ) 
        slope, intercept, r, p, std_err = stats.linregress(model_predictions_AP_hex_one, actual_AP)
        print("AP model R: " + str(r)) 
        print("AP model corrcoef: " + str(np.corrcoef(model_predictions_AP_hex_one, actual_AP)[0][1])) 
        print("AP model spearmanr: " + str(stats.spearmanr(model_predictions_AP_hex_one, actual_AP)[0])) 
        print("AP model R2: " + str(sklearn.metrics.r2_score(model_predictions_AP_hex_one, actual_AP)))  
        mymodel = list(map(myfunc, model_predictions_AP_hex_one)) 
        plt.plot(
            model_predictions_AP_hex_one,
            mymodel, color = 'r'
        )  
        plt.savefig(hex_predictions_png_name(MY_MODEL_DATA_PATH, number, 'weak', 1))
        plt.close()

    print("Seed " + str(seed) + " All")

    slope, intercept, r, p, std_err = stats.linregress(model_predictions_seq_hex_seed[seed], actual_AP_mid)
    print("Seq. model R: " + str(r))
    print("Seq. model corrcoef: " + str(np.corrcoef(model_predictions_seq_hex_seed[seed], actual_AP_mid)[0][1])) 
    print("Seq. model spearmanr: " + str(stats.spearmanr(model_predictions_seq_hex_seed[seed], actual_AP_mid)[0])) 
    print("Seq. model R2: " + str(sklearn.metrics.r2_score(model_predictions_seq_hex_seed[seed], actual_AP_mid)))

    plt.figure()
    plt.title(
        merge_type_iteration(SEQ_MODEL_DATA_PATH, 'weak', 1, 0).replace("Test 0 Weak 1", "Seed " + str(seed)) 
        + "\nPredicted self assembly probability for hexapeptides"
    )
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("AP") 
    plt.scatter(
        model_predictions_seq_hex_seed[seed],
        actual_AP_mid,
        label = "Seed " + str(seed)
    )   
    mymodel = list(map(myfunc, model_predictions_seq_hex_seed[seed])) 
    plt.plot(
        model_predictions_seq_hex_seed[seed],
        mymodel, color = 'r'
    )   
    plt.savefig(hex_predictions_seed_name(SEQ_MODEL_DATA_PATH))
    plt.close()

    slope, intercept, r, p, std_err = stats.linregress(model_predictions_hex_seed[seed], actual_AP_mid)
    print("Seq. and AP model R: " + str(r))
    print("Seq. and AP model corrcoef: " + str(np.corrcoef(model_predictions_hex_seed[seed], actual_AP_mid)[0][1])) 
    print("Seq. and AP model spearmanr: " + str(stats.spearmanr(model_predictions_hex_seed[seed], actual_AP_mid)[0])) 
    print("Seq. and AP model R2: " + str(sklearn.metrics.r2_score(model_predictions_hex_seed[seed], actual_AP_mid)))

    plt.figure()
    plt.title(
        merge_type_iteration(MODEL_DATA_PATH, 'weak', 1, 0).replace("Test 0 Weak 1", "Seed " + str(seed)) 
        + "\nPredicted self assembly probability for hexapeptides"
    )
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("AP") 
    plt.scatter(
        model_predictions_hex_seed[seed],
        actual_AP_mid,
        label = "Seed " + str(seed)
    )   
    mymodel = list(map(myfunc, model_predictions_hex_seed[seed])) 
    plt.plot(
        model_predictions_hex_seed[seed],
        mymodel, color = 'r'
    )   
    plt.savefig(hex_predictions_seed_name(MODEL_DATA_PATH))
    plt.close()

    slope, intercept, r, p, std_err = stats.linregress(model_predictions_AP_hex_seed[seed], actual_AP_mid)
    print("AP model R: " + str(r))
    print("AP model corrcoef: " + str(np.corrcoef(model_predictions_AP_hex_seed[seed], actual_AP_mid)[0][1])) 
    print("AP model spearmanr: " + str(stats.spearmanr(model_predictions_AP_hex_seed[seed], actual_AP_mid)[0])) 
    print("AP model R2: " + str(sklearn.metrics.r2_score(model_predictions_AP_hex_seed[seed], actual_AP_mid)))

    plt.figure()
    plt.title(
        merge_type_iteration(MY_MODEL_DATA_PATH, 'weak', 1, 0).replace("Test 0 Weak 1", "Seed " + str(seed)) 
        + "\nPredicted self assembly probability for hexapeptides"
    )
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("AP") 
    plt.scatter(
        model_predictions_AP_hex_seed[seed],
        actual_AP_mid,
        label = "Seed " + str(seed)
    )   
    mymodel = list(map(myfunc, model_predictions_AP_hex_seed[seed])) 
    plt.plot(
        model_predictions_AP_hex_seed[seed],
        mymodel, color = 'r'
    )   
    plt.savefig(hex_predictions_seed_name(MY_MODEL_DATA_PATH))
    plt.close()
   
print("All seeds") 

plt.figure()
plt.title(
    merge_type_iteration(SEQ_MODEL_DATA_PATH, 'weak', 1, 0).replace("Test 0 Weak 1", "All tests") 
    + "\nPredicted self assembly probability for hexapeptides"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
for seed in seed_list:
    plt.scatter(
        model_predictions_seq_hex_seed[seed],
        actual_AP_mid,
        label = "Seed " + str(seed)
    ) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_seq_hex, actual_AP_long)
print("Seq. model R: " + str(r))
print("Seq. model corrcoef: " + str(np.corrcoef(model_predictions_seq_hex, actual_AP_long)[0][1])) 
print("Seq. model spearmanr: " + str(stats.spearmanr(model_predictions_seq_hex, actual_AP_long)[0])) 
print("Seq. model R2: " + str(sklearn.metrics.r2_score(model_predictions_seq_hex, actual_AP_long)))
mymodel = list(map(myfunc, model_predictions_seq_hex)) 
plt.plot(
    model_predictions_seq_hex,
    mymodel, color = 'b'
)  
plt.legend()
plt.savefig(hex_predictions_final_name(SEQ_MODEL_DATA_PATH))
plt.close()

plt.figure()
plt.title(
    merge_type_iteration(MODEL_DATA_PATH, 'weak', 1, 0).replace("Test 0 Weak 1", "All tests")
    + "\nPredicted self assembly probability for hexapeptides"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
for seed in seed_list:
    plt.scatter(
        model_predictions_hex_seed[seed],
        actual_AP_mid,
        label = "Seed " + str(seed)
    ) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_hex, actual_AP_long)
print("Seq. and AP model R: " + str(r)) 
print("Seq. and AP model corrcoef: " + str(np.corrcoef(model_predictions_hex, actual_AP_long)[0][1])) 
print("Seq. and AP model spearmanr: " + str(stats.spearmanr(model_predictions_hex, actual_AP_long)[0])) 
print("Seq. and AP model R2: " + str(sklearn.metrics.r2_score(model_predictions_hex, actual_AP_long)))  
mymodel = list(map(myfunc, model_predictions_hex)) 
plt.plot(
    model_predictions_hex,
    mymodel, color = 'b'
)  
plt.legend()
plt.savefig(hex_predictions_final_name(MODEL_DATA_PATH))
plt.close() 

plt.figure()
plt.title(
    merge_type_iteration(MY_MODEL_DATA_PATH, 'weak', 1, 0).replace("Test 0 Weak 1", "All tests")
    + "\nPredicted self assembly probability for hexapeptides"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
for seed in seed_list:
    plt.scatter(
        model_predictions_AP_hex_seed[seed],
        actual_AP_mid,
        label = "Seed " + str(seed)
    ) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_AP_hex, actual_AP_long)
print("AP model R: " + str(r)) 
print("AP model corrcoef: " + str(np.corrcoef(model_predictions_AP_hex, actual_AP_long)[0][1])) 
print("AP model spearmanr: " + str(stats.spearmanr(model_predictions_AP_hex, actual_AP_long)[0])) 
print("AP model R2: " + str(sklearn.metrics.r2_score(model_predictions_AP_hex, actual_AP_long)))  
mymodel = list(map(myfunc, model_predictions_AP_hex)) 
plt.plot(
    model_predictions_AP_hex,
    mymodel, color = 'b'
)  
plt.legend()
plt.savefig(hex_predictions_final_name(MY_MODEL_DATA_PATH))
plt.close()  