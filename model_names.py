from model import ParallelPrediction, ParallelPrediction_LSTM, GIN, GraphSAGE, GraphSAGE_LSTM

model_names_dict = {'parallel_prediction': ParallelPrediction, 'parallel_prediction_lstm': ParallelPrediction_LSTM,
                    'gin_parallel': GIN, 'sage_parallel': GraphSAGE, 'sage_lstm': GraphSAGE_LSTM}
