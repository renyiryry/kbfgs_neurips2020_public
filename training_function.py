import datetime
import pytz

from datasets_utils import *
from models import *

from kbfgs_utils import *
from kfac_utils import *
from first_order_methods import *

from functions_utils import *
from training_utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train_model(
    dataset_name = 'MNIST', # ['MNIST', 'FACES']
    home_path = '/home/jupyter/',
    algorithm = 'K-BFGS', # ['KFAC', 'K-BFGS', 'K-BFGS(L)', 'RMSprop', 'Adam', 'SGD-momentum']
    lr = 0.3,
    lambda_damping = 0.3,
    batch_size = 1000,
    RMSprop_epsilon = 1e-4,
    if_gpu = True,
    if_max_epoch = 0,
    max_epoch = 100,
    max_cpu_time = 500,
    verbose = True):
    
    args = {}
    args['algorithm'] = algorithm
    args['alpha'] = lr
    args['if_max_epoch'] = if_max_epoch
    args['momentum_gradient_rho'] = 0.9
    
    args['dataset'] = dataset_name
    
    args['N1'] = batch_size
    args['N2'] = args['N1']
    
    if args['dataset'] in ['MNIST', 'CURVES']:
        args['name_loss'] = 'logistic-regression-sum-loss'
    elif args['dataset'] == 'FACES':
        args['name_loss'] = 'linear-regression-half-MSE'
        
    args['tau'] = 10**(-5) 
    
    if algorithm == 'KFAC':
        args['kfac_rho'] = 0.9
        args['kfac_inverse_update_freq'] = 20
        args['kfac_damping_lambda'] = lambda_damping
        args['matrix_name'] = 'Fisher'
        args['if_second_order_algorithm'] = True
    
    elif algorithm in ['K-BFGS', 'K-BFGS(L)']:
        args['Kron_BFGS_A_decay'] = 0.9
        args['Kron_BFGS_A_inv_freq'] = 20

        args['Kron_BFGS_H_initial'] = 1
        args['Kron_LBFGS_Hg_initial'] = 1

        args['Kron_BFGS_A_LM_epsilon'] = np.sqrt(lambda_damping)
        args['Kron_BFGS_H_epsilon'] = np.sqrt(lambda_damping)

        args['Kron_BFGS_number_s_y'] = 100
        args['matrix_name'] = 'EF'
        args['if_second_order_algorithm'] = True
        
    elif algorithm in ['RMSprop', 'Adam']: 
        args['RMSprop_epsilon'] = RMSprop_epsilon
        args['matrix_name'] = 'None'
        args['if_second_order_algorithm'] = False
    else : 
        args['matrix_name'] = 'None'
        args['if_second_order_algorithm'] = False
        
    
    args['record_epoch'] = 1
    
    args['home_path'] = home_path # gcp
    args['if_gpu'] = if_gpu
    args['tuning_criterion'] = 'train_loss'
    args['seed_number'] = 9999
    
    
    params = {}
    torch.cuda.empty_cache()
    seed_number = args['seed_number']
    params['seed_number'] = seed_number

    np.random.seed(seed_number)
    torch.manual_seed(seed_number)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    params['home_path'] = args['home_path']
    params['if_gpu'] = args['if_gpu']

    if params['if_gpu'] and torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 

    device = torch.device(dev)  
    params['device'] = device

    params['algorithm'] = args['algorithm']
    params['matrix_name'] = args['matrix_name']

    if_max_epoch = args['if_max_epoch'] # 0 means max_time
    if if_max_epoch:
        args['max_epoch/time'] = max_epoch
        max_epoch = max_epoch
    else:
        args['max_epoch/time'] = max_cpu_time
        max_time = max_cpu_time

    record_epoch = args['record_epoch']

    params['name_dataset'] = args['dataset']
    params['name_loss'] = args['name_loss']
    params['momentum_gradient_rho'] = args['momentum_gradient_rho']
    
    #########################
    model = get_model(params)
    #########################
    
    params['name_model'] = model.name_model
    params['layersizes'] = model.layersizes
    params['layers_params'] = model.layers_params
    params['numlayers'] = model.numlayers
    
    if verbose:
        print('name_loss:')
        print(model.name_loss)
        print('Model created.')

    data_ = {}
    data_['model'] = model
    #########################
    dataset = read_data_sets(params['name_dataset'], params['name_model'], params['home_path'], one_hot=False)
    #########################
    
    data_['dataset'] = dataset

    X_test = dataset.test.images
    t_test = dataset.test.labels
    X_train = dataset.train.images
    t_train = dataset.train.labels

    params['num_train_data'] = len(t_train)
    params['alpha'] = args['alpha']
    params['if_second_order_algorithm'] = args['if_second_order_algorithm']
    params['tau'] = args['tau']
    
    #########################
    data_, params = train_initialization(data_, params, args)
    #########################
    
    data_['model_grad_momentum'] = get_zero_torch(params)

    epochs = [0]
    timesCPU = [0]

    train_losses = []
    train_acces = []

    test_acces = []
    test_losses = []

    reduction = 'mean'

    test_loss_0, test_acc_0 = get_regularized_loss_and_acc_from_x_whole_dataset(model, X_test, t_test,reduction, params)
    test_losses.append(test_loss_0)
    test_acces.append(test_acc_0)

    loss_0, acc_0 = get_regularized_loss_and_acc_from_x_whole_dataset(model, X_train, t_train, reduction, params)
    train_losses.append(loss_0)
    train_acces.append(acc_0)

    N1 = params['N1']
    iter_per_epoch = int(len(t_train) / N1)
    iter_per_record = int(np.floor(len(t_train) * record_epoch / N1))

    # Training
    print('Begin training...')
    epoch = -1
    i = -1

    while not get_if_stop(args, i+1, iter_per_epoch, timesCPU):
        i += 1
        params['i'] = i

        if i % iter_per_record == 0:
            start_time_wall_clock = time.time()
            start_time_cpu = time.process_time()
            epoch += 1

        # get minibatch
        X_mb, t_mb = dataset.train.next_batch(N1)
        X_mb = torch.from_numpy(X_mb).to(device)
        t_mb = torch.from_numpy(t_mb).to(device)

        # Forward
        z, a, h = model.forward(X_mb)

        reduction = 'mean'
        loss = get_loss_from_z(model, z, t_mb, reduction)

        # backward and gradient
        model.zero_grad()
        loss.backward()

        model_grad_torch = get_model_grad(model, params)
        model_grad_torch = get_plus_torch(model_grad_torch,get_multiply_scalar_no_grad(params['tau'], model.layers_weight))
        data_['model_grad_torch'] = model_grad_torch

        if get_if_nan(model_grad_torch):
            print('Error: nan in model_grad_torch')
            break

        rho = params['momentum_gradient_rho']
        data_['model_grad_momentum'] = get_plus_torch(get_multiply_scalar(rho, data_['model_grad_momentum']),get_multiply_scalar(1 - rho, model_grad_torch))
        data_['model_grad_used_torch'] = data_['model_grad_momentum']

        # get second order caches
        data_['X_mb'] = X_mb
        data_['t_mb'] = t_mb

        data_ = get_second_order_caches(z, a, h, data_, params)

        model = data_['model']
        algorithm = params['algorithm']

        if algorithm == 'KFAC':    
            data_, params = kfac_update(data_, params)
        elif algorithm == 'SGD-momentum':
            data_ = SGD_update(data_, params)
        elif algorithm in ['RMSprop', 'Adam']:
            data_ = RMSprop_update(data_, params)
        elif algorithm in ['K-BFGS', 'K-BFGS(L)']:
            data_, params = Kron_BFGS_update(data_, params)

        p_torch = data_['p_torch']

        if get_if_nan(p_torch):
            print('Error: nan in p_torch')
            break

        model = update_parameter(p_torch, model, params)

        if get_if_nan(model.layers_weight):
            print('Error: nan in model.layers_weight')
            break

        if (i+1) % iter_per_record == 0:
            my_date = datetime.datetime.now(pytz.timezone('US/Eastern'))
            my_date = my_date.strftime("%d/%m/%Y %H:%M:%S")

            timesCPU_i = time.process_time() - start_time_cpu

            reduction = 'mean'
            loss_i, acc_i = get_regularized_loss_and_acc_from_x_whole_dataset(model, X_train, t_train, reduction, params)

            if math.isnan(loss_i):
                print('Warning: loss_i is NAN.')
                break

            timesCPU.append(timesCPU_i)
            
            if epoch > 0:
                timesCPU[-1] = timesCPU[-1] + timesCPU[-2]

            train_losses.append(loss_i)
            train_acces.append(acc_i)

            reduction = 'mean'
            test_loss_i, test_acc_i = get_regularized_loss_and_acc_from_x_whole_dataset(model, X_test, t_test, reduction, params)
            test_losses.append(test_loss_i)
            test_acces.append(test_acc_i)

            epochs.append((epoch + 1) * record_epoch)
            
            if verbose :
                print('Learning rate: {0:.5f}'.format(params['alpha']))
                print('Epoch-{0:.3f}'.format(epochs[-1]))

                print('Training loss: {0:.3f}'.format(train_losses[-1]))
                print('Training accuracy: {0:.3f}'.format(train_acces[-1]))

                print('Testing loss: {0:.3f}'.format(test_losses[-1]))
                print('Testing accuracy: {0:.3f}'.format(test_acces[-1]))
                print('\n')

    epochs = np.asarray(epochs)
    timesCPU = np.asarray(timesCPU)
    train_losses = np.asarray(train_losses)
    train_acces = np.asarray(train_acces)
    test_losses = np.asarray(test_losses)
    test_acces = np.asarray(test_acces)
    
    dict_result = {'algorithm' : params['algorithm'],
                   'dataset' : dataset_name,
                   'train_losses': train_losses,
                   'train_acces': train_acces,
                   'test_losses': test_losses,
                   'test_acces': test_acces,
                   'timesCPU': timesCPU,
                   'epochs': epochs}
    return dict_result
  
