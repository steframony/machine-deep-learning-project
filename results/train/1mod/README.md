# Modifica 1
Modificato il GraphConvLayer

```
class GraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super(GraphConvLayer, self).__init__()
        self.mlp = nn.Linear(in_feats * 2, out_feats, bias=bias)

    def forward(self, bipartite, feat):
        if isinstance(feat, tuple):
            srcfeat, dstfeat = feat
        else:
            srcfeat = feat
            dstfeat = feat[: bipartite.num_dst_nodes()]
        graph = bipartite.local_var()

        graph.srcdata["h"] = srcfeat
        
        graph.update_all(
            fn.u_mul_e("h", "affine", "m"), fn.sum(msg="m", out="h")
        )

        gcn_feat = torch.cat([dstfeat, graph.dstdata["h"]], dim=-1)
        out = self.mlp(gcn_feat)
        return out
```

Ho modificato il modo in cui viene invocato il update_all, nel seguente modo:

```
graph.update_all(
    fn.u_mul_e("h", "raw_affine", "m"), 
    fn.sum(msg="m", out="h")
)
```
Questo cosa comporta?
