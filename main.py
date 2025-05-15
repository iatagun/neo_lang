import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Policy network: Embedding + GRU + Linear
class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, hidden=None):
        embedded = self.embedding(input_seq)
        outputs, hidden = self.gru(embedded, hidden)
        logits = self.fc(outputs)
        return logits, hidden

# Reinforcement Learning Chat Agent using REINFORCE
class ReinforceChatAgent:
    def __init__(self, vocab, embed_size=128, hidden_size=256, lr=1e-3):
        # vocab: dict mapping token->index
        self.vocab = vocab
        self.rev_vocab = {i: w for w, i in vocab.items()}
        self.policy = PolicyNetwork(len(vocab), embed_size, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.reset_episode()

    def reset_episode(self):
        self.log_probs = []  # store log probabilities for each action
        self.rewards = []    # store rewards for each token

    def select_action(self, state_seq):
        # state_seq: list of tokens (strings)
        idxs = [self.vocab.get(tok, self.vocab['<unk>']) for tok in state_seq]
        input_tensor = torch.tensor([idxs])  # shape (1, seq_len)
        logits, _ = self.policy(input_tensor)
        probs = torch.softmax(logits[0, -1], dim=-1)
        dist = Categorical(probs)
        action_idx = dist.sample()
        self.log_probs.append(dist.log_prob(action_idx))
        return self.rev_vocab.get(action_idx.item(), '<unk>')

    def generate_response(self, input_text, max_len=20):
        tokens = input_text.strip().split()
        response = []
        for _ in range(max_len):
            next_word = self.select_action(tokens + response)
            if next_word == '<eos>':
                break
            response.append(next_word)
        return ' '.join(response)

    def finish_episode(self, gamma=0.99):
        # Compute discounted returns
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # Normalize rewards
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        # Compute policy loss
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.reset_episode()

# Interactive loop
def main():
    # Example vocabulary; expand for real use
    vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2, 'hello': 3, 'how': 4, 'are': 5, 'you': 6}
    agent = ReinforceChatAgent(vocab)
    print("Reinforcement Learning Chat Agent ready. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        reply = agent.generate_response(user_input)
        print(f"Agent: {reply}")
        # Ask for user feedback as reward
        try:
            r = float(input("Reward (e.g. 1.0 positive, 0.0 negative): "))
        except ValueError:
            r = 0.0
        agent.rewards.append(r)
        agent.finish_episode()

if __name__ == '__main__':
    main()
