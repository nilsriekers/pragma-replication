import signal
import yaml
import util
from main import main_wrapper
from multiprocessing import Pool

CONFIG = """
opt:
    epochs: 10
    batch_size: 100
    alternatives: 1

    rho: 0.95
    eps: 0.000001
    lr: 1
    clip: 10

model:
    prop_embedding_size: 50
    word_embedding_size: 50
    hidden_size: 100
"""

def handle_interrupt(signal, frame):
    print("KeyboardInterrupt detected. Terminating...")
    raise KeyboardInterrupt

def train_model(hyperparams):
    hidden_size, batch_size, lr = hyperparams
    
    # Load the configuration from the CONFIG string
    config = util.Struct(**yaml.load(CONFIG))

    # Update the configuration
    config.opt.batch_size = batch_size
    config.opt.lr = lr
    config.model.hidden_size = hidden_size

    # Call main_wrapper() with the updated configuration
    model_name = "model_h{}_b{}_lr{}".format(hidden_size, batch_size, lr)
    main_wrapper("train.base", "abstract", config, model_name)

# Create a function to run the training using multiprocessing
def run_training(hidden_size_list, batch_size_list, lr_list):
    signal.signal(signal.SIGINT, handle_interrupt)
    p = Pool()
    try:
        for _ in p.imap(train_model, [(hidden_size, batch_size, lr) for hidden_size in hidden_size_list for batch_size in batch_size_list for lr in lr_list]):
            pass
    except KeyboardInterrupt:
        p.terminate()
        p.close()
    except Exception:
        p.terminate()
    else:
        p.close()
    p.join()

# Call the run_training() function
if __name__ == "__main__":
    hidden_size_list = [100] #[50, 100, 150]
    batch_size_list = [100] #[50, 100, 150]
    lr_list = [1.0] #[0.1, 0.5, 1.0]
    run_training(hidden_size_list, batch_size_list, lr_list)
