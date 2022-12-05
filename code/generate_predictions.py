from utils import MODEL_DATA_PATH, SEQ_MODEL_DATA_PATH, MY_MODEL_DATA_PATH
from after_training import after_training, after_training_seq, after_training_AP, no_file_after_training, no_file_after_training_seq, no_file_after_training_AP, model_predict, model_predict_seq, model_predict_AP, no_file_model_predict, no_file_model_predict_seq, no_file_model_predict_AP
from custom_plots import make_ROC_plots, make_PR_plots, output_metrics, hist_predicted

def generate_predictions(num_props, best_batch_size, best_model_file, test_number, test_data, test_labels, properties, names = ['AP'], offset = 1, masking_value = 2): 
     # Get predictions from all the models for data that was labeled beforehand
     model_predictions = model_predict(num_props, best_batch_size, test_data, test_labels, best_model_file) 
      
     #Plot ROC curves for all models
     make_ROC_plots(MODEL_DATA_PATH, test_number, test_labels, model_predictions=model_predictions)
     
     #Plot PR curves for all models
     make_PR_plots(MODEL_DATA_PATH, test_number, test_labels, model_predictions=model_predictions)
    
     # Output adjusted accuracy, F1 score and ROC AUC score for all models
     output_metrics(test_labels, model_predictions) 
     
     # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
     hist_predicted(MODEL_DATA_PATH, test_number, test_labels, model_predictions) 
    
     # Generate predictions on data that has no label beforehand
     after_training(num_props, best_batch_size, test_number, best_model_file, properties, names, offset, masking_value)
     
def generate_predictions_seq(best_batch_size, best_model_file, test_number, test_data, test_labels, properties, names = ['AP'], offset = 1, masking_value = 2): 
     # Get predictions from all the models for data that was labeled beforehand
     model_predictions = model_predict_seq(best_batch_size, test_data, test_labels, best_model_file) 
      
     #Plot ROC curves for all models
     make_ROC_plots(SEQ_MODEL_DATA_PATH, test_number, test_labels, model_predictions=model_predictions)
     
     #Plot PR curves for all models
     make_PR_plots(SEQ_MODEL_DATA_PATH, test_number, test_labels, model_predictions=model_predictions)
    
     # Output adjusted accuracy, F1 score and ROC AUC score for all models
     output_metrics(test_labels, model_predictions) 
     
     # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
     hist_predicted(SEQ_MODEL_DATA_PATH, test_number, test_labels, model_predictions) 
    
     # Generate predictions on data that has no label beforehand
     after_training_seq(best_batch_size, test_number, best_model_file, properties, names, offset, masking_value)

def generate_predictions_AP(num_props, best_batch_size, best_model_file, test_number, test_data, test_labels, properties, names = ['AP'], offset = 1, masking_value = 2): 
    # Get predictions from all the models for data that was labeled beforehand
    model_predictions = model_predict_AP(num_props, best_batch_size, test_data, test_labels, best_model_file) 
     
    #Plot ROC curves for all models
    make_ROC_plots(MY_MODEL_DATA_PATH, test_number, test_labels, model_predictions=model_predictions)
    
    #Plot PR curves for all models
    make_PR_plots(MY_MODEL_DATA_PATH, test_number, test_labels, model_predictions=model_predictions)
   
    # Output adjusted accuracy, F1 score and ROC AUC score for all models
    output_metrics(test_labels, model_predictions) 
    
    # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
    hist_predicted(MY_MODEL_DATA_PATH, test_number, test_labels, model_predictions) 
   
    # Generate predictions on data that has no label beforehand
    after_training_AP(num_props, best_batch_size, test_number, best_model_file, properties, names, offset, masking_value)
    
def adaboost_generate_predictions(models, alpha_values, num_props, best_batch_size, test_number, iteration, test_data, test_labels, properties, names = ['AP'], offset = 1, masking_value = 2): 
     # Get predictions from all the models for data that was labeled beforehand
     model_predictions = no_file_model_predict(num_props, best_batch_size, test_data, test_labels, models, alpha_values) 
      
     #Plot ROC curves for all models
     make_ROC_plots(MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=model_predictions)
     
     #Plot PR curves for all models
     make_PR_plots(MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=model_predictions)
    
     # Output adjusted accuracy, F1 score and ROC AUC score for all models
     output_metrics(test_labels, model_predictions) 
     
     # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
     hist_predicted(MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions) 
    
     # Generate predictions on data that has no label beforehand
     no_file_after_training(num_props, best_batch_size, test_number, iteration, models, alpha_values, properties, names, offset, masking_value)
     
def adaboost_generate_predictions_seq(models, alpha_values, best_batch_size, test_number, iteration, test_data, test_labels, properties, names = ['AP'], offset = 1, masking_value = 2): 
     # Get predictions from all the models for data that was labeled beforehand
     model_predictions = no_file_model_predict_seq(best_batch_size, test_data, test_labels, models, alpha_values) 
      
     #Plot ROC curves for all models
     make_ROC_plots(SEQ_MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=model_predictions)
     
     #Plot PR curves for all models
     make_PR_plots(SEQ_MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=model_predictions)
    
     # Output adjusted accuracy, F1 score and ROC AUC score for all models
     output_metrics(test_labels, model_predictions) 
     
     # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
     hist_predicted(SEQ_MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions) 
    
     # Generate predictions on data that has no label beforehand
     no_file_after_training_seq(best_batch_size, test_number, iteration, models, alpha_values, properties, names, offset, masking_value)

def adaboost_generate_predictions_AP(models, alpha_values, num_props, best_batch_size, test_number, iteration, test_data, test_labels, properties, names = ['AP'], offset = 1, masking_value = 2): 
    # Get predictions from all the models for data that was labeled beforehand
    model_predictions = no_file_model_predict_AP(num_props, best_batch_size, test_data, test_labels, models, alpha_values) 
     
    #Plot ROC curves for all models
    make_ROC_plots(MY_MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=model_predictions)
    
    #Plot PR curves for all models
    make_PR_plots(MY_MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=model_predictions)
   
    # Output adjusted accuracy, F1 score and ROC AUC score for all models
    output_metrics(test_labels, model_predictions) 
    
    # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
    hist_predicted(MY_MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions) 
   
    # Generate predictions on data that has no label beforehand
    no_file_after_training_AP(num_props, best_batch_size, test_number, iteration, models, alpha_values, properties, names, offset, masking_value)