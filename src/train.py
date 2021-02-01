from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, RMSprop
from torch.autograd import grad
from torch import norm, cat, no_grad, save, rand, stack, ones_like
from torch import abs as t_abs
from torch.nn import BCEWithLogitsLoss
from numpy import inf as np_inf
from tqdm import tqdm
import evaluate
def train_wdgrl(model, critic, criterion,
                n_epochs, loader_s, loader_t,
                save_name, device='cpu', patience=2, 
                n_critic_iters=6, n_discriminator_iters=10,
                ws_scaler=1, gp_scaler=1, lr_critic=5e-5,
                lr_discriminator=1e-4):
    
    model.train()

    prev_val_loss = np_inf
    
    optim_critic = RMSprop(critic.parameters(), lr=lr_critic)
    optim_discriminator = Adam(model.parameters(), lr=lr_discriminator)
    lr_schedule = ReduceLROnPlateau(optim_discriminator, patience=patience, 
                                    verbose=True,  factor=0.5)
    for epoch in range(1, n_epochs + 1):

        batches = zip(loader_s, loader_t)
        n_batches = min(len(loader_s), len(loader_t))
        
        total_loss_critic = 0
        total_accuracy = 0
        total_wasserstein_distance = 0
        total_wasserstein_distance_clf = 0
        total_discriminator_loss = 0

        for (source_fatures, source_labels, _), (target_fatures, _, _) in tqdm(batches, leave=False, total=n_batches):
            
            set_requires_grad(model.feature_extractor, False) # Disable training of feature_extractor
            set_requires_grad(critic, True) # Enable training of critic
                
            source_fatures, target_fatures = source_fatures.to(device), target_fatures.to(device) 
            source_labels = source_labels.to(device)
            
            # Get distribution of source and target domain samples in hidden space
            with no_grad():
                h_s = model.feature_extractor(source_fatures).data.view(source_fatures.shape[0], -1)
                h_t = model.feature_extractor(target_fatures).data.view(target_fatures.shape[0], -1)
                
            
            # Critic training loop
            for _ in range(n_critic_iters):
                
                # Compute Wasserstein distance between source and target domain distribution
                critic_source = critic(h_s)
                critic_target = critic(h_t)
                wasserstein_distance = ws_scaler*(critic_source.mean() 
                                            - critic_target.mean())

                # Compute gradient penalty (1-Lipschitz constrain)
                gp = gp_scaler*lipschitz_constrain(critic, h_s, h_t, device)

                critic_loss = -t_abs(wasserstein_distance) + gp

                optim_critic.zero_grad()
                critic_loss.backward()
                optim_critic.step()

                total_loss_critic += critic_loss.item()
                total_wasserstein_distance += wasserstein_distance
                
            # Discriminator trianing loop
            set_requires_grad(model.feature_extractor, True) # Enable training of feature_extractor
            set_requires_grad(critic, False) # Disable training of critic
            
            for _ in range(n_discriminator_iters):
                h_s = model.feature_extractor(source_fatures).view(source_fatures.shape[0], -1)
                h_t = model.feature_extractor(target_fatures).view(target_fatures.shape[0], -1)

                predicted_labels = model.discriminator(h_s)
                discriminator_loss = criterion(predicted_labels.view(-1), source_labels)
                wasserstein_distance = ws_scaler*(critic(h_s).mean() 
                                                    - critic(h_t).mean())

                combined_loss = discriminator_loss + t_abs(wasserstein_distance)
                total_discriminator_loss += discriminator_loss
                total_wasserstein_distance_clf += wasserstein_distance
                
                optim_discriminator.zero_grad()
                combined_loss.backward()
                optim_discriminator.step()
            

        lr_schedule.step(combined_loss)

        # Compute mean losses
        mean_loss = total_loss_critic / (n_batches * n_critic_iters) 
        mean_wasserstein_distance = total_wasserstein_distance / (n_batches * n_critic_iters) 
        
        total_wasserstein_distance_clf = total_wasserstein_distance_clf / (n_batches * n_discriminator_iters) 
        mean_clf_loss = total_discriminator_loss / (n_batches * n_discriminator_iters) 
        total_loss_discriminator = total_wasserstein_distance_clf + mean_clf_loss

        tqdm.write(f'EPOCH {epoch:03d}: total_loss_critic={mean_loss:.4f}, ' f'mean_ws_dist_critic={mean_wasserstein_distance:.4f}, ' 
                f'clf_loss={mean_clf_loss:.4f}, ' f'total_discriminator_loss={total_loss_discriminator:.4f}, ' 
                f'mean_ws_dist_discriminator={total_wasserstein_distance_clf:.4f}')

        # Check quality
        out, true_l = evaluate.propagte_data_through_network(loader_t, model, device)                                                    
        out_prod = out.view(-1).cpu().numpy()
        true_l = true_l.cpu().numpy()

        _ = evaluate.evaluate_model_ova(out_prod, true_l)

        if mean_clf_loss < prev_val_loss:
            prev_val_loss = mean_clf_loss
            print('Saving model ...')
            save(model.state_dict(), save_name + '.pt')

        model.eval()

    model.eval()

def train_nn(model, lr, n_epochs, train_data_loader, 
            valid_data_loader, save_name, device='cpu', patience=4):
    
    optim = Adam(model.parameters(), lr)
    lr_schedule = ReduceLROnPlateau(optim, patience=patience, verbose=True, factor=0.5)

    prev_val_loss = np_inf

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss, train_accuracy = do_epoch_nn(model, train_data_loader, device, optim=optim)
        model.eval()
        with no_grad():
            val_loss, val_accuracy = do_epoch_nn(model, valid_data_loader, device, optim=None)

        tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        if val_loss < prev_val_loss:
            prev_val_loss = val_loss
            print('Saving model ...')
            save(model.state_dict(), save_name + '.pt')

        lr_schedule.step(val_loss)
    model.eval()

def do_epoch_nn(model, dataloader, device, optim=None):
    total_loss = 0
    total_accuracy = 0
    for x, y_true, weights in tqdm(dataloader, leave=False):
        x, y_true, weights = x.to(device), y_true.to(device), weights.to(device)
        criterion = BCEWithLogitsLoss(pos_weight=weights)
        criterion.weights = weights
        y_pred = model(x)
        loss = criterion(y_pred.view(-1), y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        l1_regularization = 0.000005 * norm(cat([x.view(-1) for x in model.parameters()]), 1)
        total_loss += loss.item() + l1_regularization
        total_accuracy += ((y_pred > 0.5) == y_true).float().mean().item()

    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy

def lipschitz_constrain(discriminator, h_s, h_t, device):
    
    random_samples = rand(h_s.size(0), 1).to(device)
    distribution_diff = h_t - h_s
    interpolated_samples = h_s + (random_samples * distribution_diff)
    interpolated_samples = stack([interpolated_samples, h_s, h_t]).requires_grad_()

    preds = discriminator(interpolated_samples)
    gradients = grad(preds, interpolated_samples, grad_outputs=ones_like(preds),
                     retain_graph=True, create_graph=True)
    
    gradients = gradients[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty

def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad