import torch
from stanza.models.common import pretrain

# Path to the old pretrain file
old_pretrain_path = "./twitter-stanza/saved_models/depparse/en_tweet.pretrain.pt"

# Path to save the new format
new_pretrain_path = "./twitter-stanza/saved_models/depparse/en_tweet.pretrain_resaved.pt"

# Load the old pretrain
# weights_only=False allows loading old numpy-based format
checkpoint = torch.load(old_pretrain_path, weights_only=False)

# Save in modern Stanza format using pretrain.save
#pretrain.save(checkpoint, new_pretrain_path)
torch.save(checkpoint, new_pretrain_path)

print(f"Resaved pretrain to: {new_pretrain_path}")
