该工程主要修改了detectorn2的amp训练和sybn，

- 由于deform_conv不支持amp，所以在用到amp的地方先转化为float，具体代码如下：

```
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        if self.deform_modulated:
            offset_mask = self.conv2_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            ##debug Hanson
            if offset.requires_grad:
                offset,out = offset.float(),out.float()
                out = self.conv2(out, offset)
                out = out.half()
            else:
                out = self.conv2(out, offset)
        out = F.relu_(out)
```

- SyBN

  在DefaultTrainer类中__init__中进行如下修改：

  ```
      def __init__(self, cfg):
          """
          Args:
              cfg (CfgNode):
          """
          super().__init__()
          logger = logging.getLogger("detectron2")
          if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
              setup_logger()
          cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
  
          # Assume these objects must be constructed in this order.
          model = self.build_model(cfg)
          model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
          optimizer = self.build_optimizer(cfg, model)
          data_loader = self.build_train_loader(cfg)
  
          model = create_ddp_model(model, broadcast_buffers=False)
          self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
              model, data_loader, optimizer
          )
  ```

  

### Train

python train_net.py

