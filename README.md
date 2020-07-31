<div align="center">
 <img src="images/logo.png" width="250"> 
</div>

Research code for visualizing organization of RNN states during learning

## Usage

Execute `main() in coordinated.job`

## Example Simulations

When the mutual information between the distribution of items in slot X and B is maximal, 
the categorization problem is easily solved. 
The last hidden state of A-X-B sequences clearly approach their target locations (black triangles, one for each category).

<div align="center">
 <img src="images/hiddens2_b=item.gif" width="324"> 
</div>

However, the embeddings for items in slot X (orange) - of the same network - do not inherent the (triangular) organization that is learned at the hidden layer.
<div align="center">
 <img src="images/embeddings_b=item.gif" width="324"> 
</div>

Only when the mutual information between X and B is zero, 
do the embeddings for items in slot X (orange) inherit the same (triangular) organization that exists at the hidden layer.

<div align="center">
 <img src="images/embeddings_b=super.gif" width="324"> 
</div>

## Compatibility

Developed on Ubuntu 16.04 using Python3.7
