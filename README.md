<div align="center">
 <img src="images/logo.png" width="250"> 
</div>

Research code for visualizing organization of RNN states during learning

## Usage

Execute `main() in coordinated.job`

## Example Simulations

When mutual information between X and B slot is maximal, 
categorization problem is solved, as seen below,

<div align="center">
 <img src="images/hiddens2_b=item.gif" width="250"> 
</div>

but the embeddings do not cluster as clearly, as shown below,
<div align="center">
 <img src="images/embeddings_b=item.gif" width="250"> 
</div>

compared to when mutual information between X and B is zero, as shown below.

<div align="center">
 <img src="images/embeddings_b=super.gif" width="250"> 
</div>

## Compatibility

Developed on Ubuntu 16.04 using Python3.7
