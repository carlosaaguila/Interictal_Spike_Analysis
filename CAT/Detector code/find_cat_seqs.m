
gdf_c1 = gdf_array_cats{1};
gdf_c2 = gdf_array_cats{2};
gdf_c3 = gdf_array_cats{3};
gdf_c4 = gdf_array_cats{4};
gdf_c5 = gdf_array_cats{5};

gdf_c1=double(gdf_c1);
gdf_c2=double(gdf_c2);
gdf_c3=double(gdf_c3);
gdf_c4=double(gdf_c4);
gdf_c5=double(gdf_c5);

gdf_c1=[ gdf_c1(:,1)+1,gdf_c1(:,2) ];
gdf_c2=[ gdf_c2(:,1)+1,gdf_c2(:,2) ];
gdf_c3=[ gdf_c3(:,1)+1,gdf_c3(:,2) ];
gdf_c4=[ gdf_c4(:,1)+1,gdf_c4(:,2) ];
gdf_c5=[ gdf_c5(:,1)+1,gdf_c5(:,2) ];

gdf_c1 = sortrows(gdf_c1,2);
gdf_c2 = sortrows(gdf_c2,2);
gdf_c3 = sortrows(gdf_c3,2);
gdf_c4 = sortrows(gdf_c4,2);
gdf_c5 = sortrows(gdf_c5,2);

[~,~,~,~,~,seqs_c1] = build_sequences(gdf_c1,nchns{1},fs);
[~,~,~,~,~,seqs_c2] = build_sequences(gdf_c2,nchns{2},fs);
[~,~,~,~,~,seqs_c3] = build_sequences(gdf_c3,nchns{3},fs);
[~,~,~,~,~,seqs_c4] = build_sequences(gdf_c4,nchns{4},fs);
[~,~,~,~,~,seqs_c5] = build_sequences(gdf_c5,nchns{5},fs);

