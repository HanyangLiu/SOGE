function genLap2( X )

      globals;
      options = [];
      options.NeighborMode = 'KNN';
      options.k = 10;
      options.WeightMode = 'HeatKernel';
      options.t = 1;
      W = constructW(X',options);
      L = diag(sum(W)) - W;
      L = 0.5*(L+L');
      save ([tmp_dir 'Laplassian.mat'], 'L');

end