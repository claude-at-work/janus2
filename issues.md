
┌──(root㉿localhost)-[~/buddy-cli/bud/scripts]
└─# python3 audit*

  Bud Index Auditor
  Preset: full — Complete audit: stats, distributions, t-SNE, clusters, similarity, co-occurrence
  Output: /root/bud-out/audit

  Loading index... OK — 1516 vectors, 1024d
  Computing statistics... OK
  Computing tag distributions... OK
  Plotting distributions... OK — 7 charts
  Plotting text length distribution... OK
  Computing co-occurrence... OK
  Plotting co-occurrence heatmaps... OK — 4 heatmaps
  Aggregating schema proposals... OK — 12 unique proposals
  Extracting vectors... OK — (1516, 1024)
  Running t-SNE (this may take a moment)... Traceback (most recent call last):
  File "/root/buddy-cli/bud/scripts/audit_index.py", line 892, in <module>
    main()
    ~~~~^^
  File "/root/buddy-cli/bud/scripts/audit_index.py", line 888, in main
    run_audit(args.index, args.out, args.preset)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/buddy-cli/bud/scripts/audit_index.py", line 798, in run_audit
    coords_2d = reduce_tsne(vectors)
  File "/root/buddy-cli/bud/scripts/audit_index.py", line 235, in reduce_tsne
    from sklearn.manifold import TSNE
ModuleNotFoundError: No module named 'sklearn'
