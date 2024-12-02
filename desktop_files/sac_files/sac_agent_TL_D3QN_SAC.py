import torch
import torch.optim as optim
import torch.nn.functional as F
from sac_networks import ActorNetwork, CriticNetwork
from sac_buffer import ReplayBuffer

class Agent():
    def __init__(self, state_size, action_size, actor_lr, critic_lr, gamma, fixed_alpha, tau, batch_size, buffer_size, hidden_size, pretrained_path=None, action_prior="uniform"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.fixed_alpha = 0.1#None
        self.tau = tau
        self.batch_size = batch_size
        
        self.target_entropy = -1*action_size  # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=actor_lr) 
        self._action_prior = action_prior
        self.pretrained_path = pretrained_path
        
        # Actor Network 
        self.actor_local = ActorNetwork(state_size, action_size, self.device, hidden_size, pretrained_path = self.pretrained_path).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)     
        
        # Critic Network (w/ Target Network)
        self.critic1 = CriticNetwork(state_size, action_size, self.device, hidden_size, pretrained_path = self.pretrained_path).to(self.device)
        self.critic2 = CriticNetwork(state_size, action_size, self.device, hidden_size, pretrained_path = self.pretrained_path).to(self.device)
        
        self.critic1_target = CriticNetwork(state_size, action_size, self.device, hidden_size, pretrained_path = self.pretrained_path).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = CriticNetwork(state_size, action_size, self.device, hidden_size, pretrained_path = self.pretrained_path).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr, weight_decay=0)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr, weight_decay=0) 

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, state_size, action_size)
        
        print("Using: ", self.device)
        
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
            
    def act(self, state):
        """Returns actions for given state as per current policy."""
        #state = torch.from_numpy(state).float().to(self.device)
        state = torch.Tensor([state]).to(self.device)
        #print('### choose action state2 = ' + str(state.shape))
        action = self.actor_local.get_action(state).detach()
        return action

    def learn(self, step, d=1):
        if self.memory.mem_cntr < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.critic1.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.critic1.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.critic1.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.critic1.device)
        dones = torch.tensor(dones, dtype=torch.uint8).to(self.critic1.device)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next = self.actor_local.evaluate(next_states)

        Q_target1_next = self.critic1_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))
        Q_target2_next = self.critic2_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))

        # take the mean of both critics for updating
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        
        if self.fixed_alpha == None:
            # Compute Q targets for current states (y_i)
            Q_targets = rewards.cpu() + (self.gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))
            print('alpha = ' + str(self.alpha))
        else:
            Q_targets = rewards.cpu() + (self.gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.fixed_alpha * log_pis_next.squeeze(0).cpu()))
        # Compute critic loss
        Q_1 = self.critic1(states, actions).cpu()
        Q_2 = self.critic2(states, actions).cpu()        
        critic1_loss = 0.5*F.mse_loss(Q_1, Q_targets.detach())
        critic2_loss = 0.5*F.mse_loss(Q_2, Q_targets.detach())
        
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        if step % d == 0:
        # ---------------------------- update actor ---------------------------- #
            if self.fixed_alpha == None:
                alpha = torch.exp(self.log_alpha)
                # Compute alpha loss
                actions_pred, log_pis = self.actor_local.evaluate(states)
                alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = alpha
                # Compute actor loss
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0
    
                actor_loss = (alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu() - policy_prior_log_probs ).mean()
            else:
                
                actions_pred, log_pis = self.actor_local.evaluate(states)
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0
    
                actor_loss = (self.fixed_alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu()- policy_prior_log_probs ).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target, self.tau)
            self.soft_update(self.critic2, self.critic2_target, self.tau)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
