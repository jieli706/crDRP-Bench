"""Suppress numerical PCA null space using training eigenvalues only."""
import numpy as np
from leakage_safe import FoldTransform as OriginalFoldTransform
class StableFoldTransform(OriginalFoldTransform):
    def fit(self,x,ids):
        super().fit(x,ids)
        ev=self.pca_.explained_variance_
        self.eigenvalue_floor_=max(1e-12,float(np.max(ev))*1e-10)
        self.active_components_=ev>self.eigenvalue_floor_
        return self
    def transform(self,x):
        # Never divide numerical null-space projections by roundoff-sized scales.
        a=self.imputer_.transform(np.asarray(x,dtype=np.float64)[:,self.valid_])[:,self.selected_]
        scores=self.pca_.transform(self.scale_.transform(a))
        out=np.zeros_like(scores)
        use=self.active_components_
        out[:,use]=(scores[:,use]-self.post_.mean_[use])/self.post_.scale_[use]
        return out.astype('float32')
