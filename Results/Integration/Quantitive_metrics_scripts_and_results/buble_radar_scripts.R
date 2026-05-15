#!/usr/bin/env Rscript
# =============================================================================
# make_integration_figures.R  -- CORRECTED VERSION
# Metric categorisation follows official scib documentation:
#   Bio  : ARI, NMI, cLISI, cell_type_ASW, silhouette, isolated_label_ASW,
#           isolated_label_F1, rare_knn_purity_mean, calinski_harabasz
#   Batch: kBET, iLISI_batch, batch_ASW, pc_regression_r2_mean,
#           graph_connectivity  <-- BATCH per scib docs, NOT bio
# Tables : PNG format, all using purple header
# Radars : overlaid = filled (alpha=0.10); multiples = filled (alpha=0.20)
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse); library(dplyr); library(tidyr)
  library(grid); library(Cairo)
})
if (!requireNamespace("fmsb", quietly = TRUE)) stop("install.packages('fmsb')")

METRICS_CSV <- "/data1/esraa/Thesis-Project/Results/Integration/Quantitive_metrics_scripts_and_results/Final_results_tables/metrics/methods_x_metrics.csv"
PERF_CSV    <- "/data1/esraa/Thesis-Project/Results/Integration/Quantitive_metrics_scripts_and_results/Final_results_tables/performance/methods_x_performance.csv"
OUTPUT_DIR  <- "/data1/esraa/Thesis-Project/Results/Integration/Quantitive_metrics_scripts_and_results/bubble_radar_plots"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
BASE_FONT <- "Helvetica"
PNG_RES   <- 180L

METHOD_NAME_MAP <- c(
  "bbknn"="BBKNN","scvi"="scVI","scanorama"="Scanorama","liger"="LIGER",
  "scanvi"="scANVI","combat"="ComBat","scgen"="scGen","harmony"="Harmony",
  "seurat"="Seurat v5","fastmnn"="fastMNN","mnn"="MNN"
)
METHOD_COLORS <- c(
  "BBKNN"="#4E79A7","scVI"="#F28E2B","Scanorama"="#59A14F","LIGER"="#E15759",
  "scANVI"="#B07AA1","ComBat"="#76B7B2","scGen"="#9C755F","Harmony"="#BAB0AC",
  "Seurat v5"="#EDC948","fastMNN"="#FF9DA7","MNN"="#4DC36A"
)
LANG <- c(
  "BBKNN"="Python","scVI"="Python","Scanorama"="Python","LIGER"="R/Python",
  "scANVI"="Python","ComBat"="R/Python","scGen"="Python","Harmony"="R/Python",
  "Seurat v5"="R","fastMNN"="R","MNN"="R/Python"
)
CATG <- c(
  "BBKNN"="Graph-based","scVI"="Deep learning",
  "Scanorama"="Manifold alignment (MNN-based)","LIGER"="Matrix factorization",
  "scANVI"="Deep learning","ComBat"="Global linear model","scGen"="Deep learning",
  "Harmony"="Linear correction (embedding)",
  "Seurat v5"="Anchor-based (RPCA)","fastMNN"="MNN-based correction",
  "MNN"="MNN-based correction"
)
OUTPUT_TYPE <- c(
  "BBKNN"="Corrected graph","scVI"="Latent embedding",
  "Scanorama"="Corrected embedding","LIGER"="Latent factors embedding",
  "scANVI"="Latent embedding","ComBat"="Corrected matrix",
  "scGen"="Corrected embedding","Harmony"="Corrected PCA embedding",
  "Seurat v5"="Corrected embedding","fastMNN"="Corrected embedding",
  "MNN"="Corrected matrix"
)



##


BIO_METRICS_BUBBLE  <- c(  "NMI",
  "ARI",
  "cLISI",
  "cell_type_ASW",
  "cell_cycle_conservation",
  "isolated_label_ASW",
  "isolated_label_F1",
  "hvg_conservation")
BIO_METRICS_RADAR   <- c(  "NMI",
  "ARI",
  "cLISI",
  "cell_type_ASW",
  "cell_cycle_conservation",
  "isolated_label_ASW",
  "isolated_label_F1",
  "hvg_conservation")
BATCH_METRICS_BUBBLE <- c( "kBET",
  "iLISI",
  "batch_ASW",
  "pcr_comparison",
  "graph_connectivity")
BATCH_METRICS_RADAR  <- c("kBET","iLISI","batch_ASW",
                           "pcr_comparison","graph_connectivity")
ALL_METRICS_RADAR    <- c(BIO_METRICS_RADAR, BATCH_METRICS_RADAR)

METRIC_DIRECTION <- c(
  "NMI"                      = TRUE,
  "ARI"                      = TRUE,
  "cLISI"                    = TRUE,
  "cell_type_ASW"            = TRUE,
  "cell_cycle_conservation"  = TRUE,
  "isolated_label_ASW"       = TRUE,
  "isolated_label_F1"        = TRUE,
  "hvg_conservation"         = TRUE,
  "trajectory_conservation"  = TRUE,

  "kBET"                     = TRUE,
  "iLISI"                    = TRUE,
  "batch_ASW"                = FALSE,  # lower = better
  "pcr_comparison"           = FALSE,  # lower = better
  "graph_connectivity"       = TRUE
)
METRIC_LABELS <- c(
  "NMI"="NMI",
  "ARI"="ARI",
  "cLISI"="cLISI",
  "cell_type_ASW"="Cell type\nASW",
  "cell_cycle_conservation"="Cell cycle\nconservation",
  "isolated_label_ASW"="Isolated label\nASW",
  "isolated_label_F1"="Isolated label\nF1",
  "hvg_conservation"="HVG\nconservation",
  "trajectory_conservation"="Trajectory\nconservation",

  "kBET"="kBET",
  "iLISI"="iLISI",
  "batch_ASW"="Batch\nASW",
  "pcr_comparison"="PCR\ncomparison",
  "graph_connectivity"="Graph\nconnectivity"
)

METRIC_LABELS_FLAT <- c(
  "NMI"="NMI",
  "ARI"="ARI",
  "cLISI"="cLISI",
  "cell_type_ASW"="Cell type ASW",
  "cell_cycle_conservation"="Cell cycle conservation",
  "isolated_label_ASW"="Isolated label ASW",
  "isolated_label_F1"="Isolated label F1",
  "hvg_conservation"="HVG conservation",
  "trajectory_conservation"="Trajectory conservation",

  "kBET"="kBET",
  "iLISI"="iLISI",
  "batch_ASW"="Batch ASW",
  "pcr_comparison"="PCR comparison",
  "graph_connectivity"="Graph connectivity",

  "perf_total_time_s"="Runtime (s)",
  "perf_peak_rss_gb"="Peak RAM (GB)"
)

# -- Palettes -----------------------------------------------------------------
# Bio palette: light (worst) -> dark navy (best)
BIO_PAL <- colorRampPalette(c(
  "#5E87F5","#4E69C3","#3D4B91","#2D2C5E","#1D0E2C"
))(100)

# Batch palette: best/lightest -> worst/darkest
BATCH_PAL <- colorRampPalette(c(
  "#BF0F35", "#B71A3C", "#AF2442", "#9F394F",
  "#873749", "#6F3542", "#57333B", "#3F3134"
))(100)

# Bio rectangle palette: pink ramp (unused, kept for reference)
BIO_RECT_PAL <- colorRampPalette(c(
  "#FFE5EC","#FFC2D1","#FFB3C6","#FF8FAB","#FB6F92"
))(100)

# Overall column rectangle palette: yellow -> deep orange (best=light, worst=dark)
# legend uses reverse=TRUE so lightest yellow = best rank at top of colour bar
OVERALL_PAL <- colorRampPalette(c(
  "#f1c71f","#FFC621","#FFB716","#FFA70F","#FF970D","#FF8711"
))(100)

BATCH_RECT_PAL <- rev(colorRampPalette(c(
  "#0a2e1e","#0d4a2f","#0f6640","#1a8a55",
  "#2daf6e","#4dcc8a","#7de0aa","#b8f0d0"
))(100))

RTPAL <- colorRampPalette(c(
  "#B5EA8C","#94BF73","#739559","#526A40"
))(100)

META_HEADER_COL    <- "#a3b7ca"
BIO_HEADER_COL   <- "#3D4B91"
BATCH_HEADER_COL <- "#9F394F"
SCALE_HEADER_COL   <- "#526A40"

# Overall column header: darkest stop of OVERALL_PAL for white-text contrast
OVERALL_HEADER_COL <- "#c56402"

# -- Data helpers -------------------------------------------------------------
clean_raw_method <- function(x) {
  x <- trimws(x)                                    # fix MNN leading whitespace
  x <- sub("_parallel\\d+_hvg_seed\\d+$", "", x)   # parallel naming variant
  x <- sub("_full_hvg_seed\\d+$",         "", x)
  x <- sub("_hvg_seed\\d+$",              "", x)
  x <- sub("_seed\\d+$",                  "", x)
  x <- sub("_full$",                      "", x)
  x <- trimws(tolower(x))
  mapped <- METHOD_NAME_MAP[x]
  ifelse(is.na(mapped), NA_character_, mapped)
}
normalize_metric <- function(vals, higher_is_better=TRUE) {
  if (higher_is_better) {
    mn <- min(vals,na.rm=TRUE); mx <- max(vals,na.rm=TRUE)
    if (!is.finite(mn)||!is.finite(mx)||mx==mn) return(rep(0.5,length(vals)))
    (vals-mn)/(mx-mn)
  } else {
    av <- abs(vals); mn <- min(av,na.rm=TRUE); mx <- max(av,na.rm=TRUE)
    if (!is.finite(mn)||!is.finite(mx)||mx==mn) return(rep(0.5,length(vals)))
    1-(av-mn)/(mx-mn)
  }
}
normalize_rank <- function(vals) {
  r <- rank(vals,ties.method="average",na.last="keep"); r/sum(!is.na(vals))
}
compute_friedman_rank <- function(norm_mat) {
  rank_mat <- apply(norm_mat,2,function(col)
    rank(-col,ties.method="average",na.last="keep"))
  rowMeans(rank_mat,na.rm=TRUE)
}

# -- Load data ----------------------------------------------------------------
message("-- Loading metrics: ", METRICS_CSV)
metrics_raw <- read_csv(METRICS_CSV, show_col_types = FALSE) %>%
  mutate(
    method       = trimws(method),
    method_clean = sapply(method, clean_raw_method)
  ) %>%
  filter(!is.na(method_clean))
message("-- Loading performance: ",PERF_CSV)
perf_raw <- read_csv(PERF_CSV,show_col_types=FALSE) %>%
  mutate(method_clean=sapply(method,clean_raw_method)) %>% filter(!is.na(method_clean))

methods_present <- union(intersect(metrics_raw$method_clean,names(METHOD_COLORS)),"MNN")
message("Methods (",length(methods_present),"): ",paste(sort(methods_present),collapse=", "))

all_avail <- intersect(ALL_METRICS_RADAR,colnames(metrics_raw))
met_df <- metrics_raw %>% filter(method_clean %in% methods_present) %>%
  select(method=method_clean,all_of(all_avail))

missing_methods <- setdiff(methods_present,met_df$method)
if (length(missing_methods)>0) {
  empty_rows <- data.frame(method=missing_methods,
    matrix(NA_real_,nrow=length(missing_methods),ncol=length(all_avail),
           dimnames=list(NULL,all_avail)),check.names=FALSE)
  met_df <- bind_rows(met_df,empty_rows)
  message("  Empty rows: ",paste(missing_methods,collapse=", "))
}

norm_df <- met_df
for (met in all_avail) {
  dir <- METRIC_DIRECTION[[met]]; if (is.null(dir)) dir <- TRUE
  norm_df[[met]] <- if (met=="calinski_harabasz") normalize_rank(met_df[[met]])
                    else normalize_metric(met_df[[met]],higher_is_better=dir)
}

bio_cols_avail   <- intersect(BIO_METRICS_BUBBLE,  all_avail)
batch_cols_avail <- intersect(BATCH_METRICS_BUBBLE,all_avail)
bio_mat   <- norm_df %>% column_to_rownames("method") %>% select(all_of(bio_cols_avail))   %>% as.matrix()
batch_mat <- norm_df %>% column_to_rownames("method") %>% select(all_of(batch_cols_avail)) %>% as.matrix()
bio_friedman   <- compute_friedman_rank(bio_mat)
batch_friedman <- compute_friedman_rank(batch_mat)

overall_rank     <- (bio_friedman+batch_friedman)/2
ranked_methods   <- names(sort(overall_rank[!is.na(overall_rank)]))
unranked_methods <- setdiff(methods_present,ranked_methods)
method_order     <- c(ranked_methods,unranked_methods)
message("Order: ",paste(method_order,collapse=" > "))

perf_df <- perf_raw %>% filter(method_clean %in% methods_present) %>%
  select(method=method_clean,perf_total_time_s,perf_peak_rss_gb) %>%
  column_to_rownames("method")

norm_wide <- norm_df %>% column_to_rownames("method")
norm_wide[is.na(norm_wide)] <- 0

# =============================================================================
# BUBBLE PLOT
# =============================================================================
make_bubble_plot <- function(out) {
  methods <- method_order; n_m <- length(methods)
  n_bio <- length(bio_cols_avail); n_batch <- length(batch_cols_avail)

  W_METHOD<-0.95;W_LANG<-0.75;W_CATG<-1.30;W_OUTPUT<-1.30
  BUB_W<-0.25;BUB_LAST<-0.38;W_OVERALL<-0.52;W_SCALE_DOT<-0.28;W_SCALE_RECT<-0.52

  col_w <- c(W_METHOD,W_LANG,W_CATG,W_OUTPUT,
             W_OVERALL,
             rep(BUB_W,n_bio-1L),BUB_LAST,
             rep(BUB_W,n_batch-1L),BUB_LAST,
             W_SCALE_RECT,W_SCALE_DOT,W_SCALE_DOT)
  cum_x <- cumsum(c(0,col_w)); TW <- sum(col_w)
  col_cx <- function(j) cum_x[j]+col_w[j]/2

  n_txt        <- 4L
  j_overall    <- n_txt+1L
  j_bio_start  <- j_overall+1L
  j_bio_end    <- j_bio_start+n_bio-1L
  j_batch_start<- j_bio_end+1L
  j_batch_end  <- j_batch_start+n_batch-1L
  j_scale_rect <- j_batch_end+1L
  j_rt_dot     <- j_scale_rect+1L
  j_mem_dot    <- j_rt_dot+1L

  x_overall_sep <- cum_x[j_overall]
  x_bio_sep     <- cum_x[j_bio_start]
  x_batch_sep   <- cum_x[j_batch_start]
  x_scale_sep   <- cum_x[j_scale_rect]
  RECT_PAD <- 0.05

  ROW_H<-0.270;HDR_H<-0.18;LBL_H<-1.80
  BAR_W_L<-0.13;BAR_SEP<-0.17;LEG_GAP<-0.52
  L_MAR<-0.22;T_MAR<-0.50;B_MAR<-LBL_H+0.60;R_MAR<-LEG_GAP+3*(BAR_W_L+BAR_SEP)+0.60
  FIG_W<-L_MAR+TW+R_MAR; FIG_H<-T_MAR+HDR_H+n_m*ROW_H+B_MAR

  pdf(out,width=FIG_W,height=FIG_H,family=BASE_FONT)
  on.exit(dev.off(),add=TRUE)
  par(mar=c(0,0,0,0),bg="white")
  plot(0,0,type="n",xlim=c(0,FIG_W),ylim=c(0,FIG_H),
       asp=NA,axes=FALSE,xlab="",ylab="",xaxs="i",yaxs="i")

  TX<-L_MAR; hdr_bot<-B_MAR+n_m*ROW_H; hdr_top<-hdr_bot+HDR_H; tbl_bot<-B_MAR
  row_top<-function(i) hdr_bot-(i-1L)*ROW_H; row_bot<-function(i) hdr_bot-i*ROW_H
  row_cy <-function(i) (row_top(i)+row_bot(i))/2
  MAX_R<-ROW_H*0.38; MIN_R<-ROW_H*0.05

  text(TX+TW/2,hdr_top+0.30,
       "scRNA-seq integration benchmark -- Bio conservation & Batch correction ranking",
       cex=0.80,font=1,col="#111111",adj=c(0.5,0.5))

  for (i in seq_len(n_m)) {
    shade<-if(i%%2==1)"white" else "#E2E2E2"
    rect(TX,row_bot(i),TX+TW,row_top(i),col=shade,border=NA)
  }
  for (y in c(hdr_top,tbl_bot)) segments(TX,y,TX+TW,y,lwd=1.6,lend=1,col="#111111")
  segments(TX,hdr_bot,TX+TW,hdr_bot,lwd=0.9,lend=1,col="#111111")
  for (sx in c(TX+x_overall_sep,TX+x_bio_sep,TX+x_batch_sep,TX+x_scale_sep))
    segments(sx,hdr_bot,sx,tbl_bot,lwd=0.4,lty=3,col="#BBBBBB")

  rect(TX,               hdr_bot,TX+x_overall_sep,hdr_top,col=META_HEADER_COL,border=NA)
  rect(TX+x_overall_sep, hdr_bot,TX+x_bio_sep,    hdr_top,col=OVERALL_HEADER_COL,border=NA)
  rect(TX+x_bio_sep,     hdr_bot,TX+x_batch_sep,  hdr_top,col=BIO_HEADER_COL, border=NA)
  rect(TX+x_batch_sep,   hdr_bot,TX+x_scale_sep,  hdr_top,col=BATCH_HEADER_COL,border=NA)
  rect(TX+x_scale_sep,   hdr_bot,TX+TW,           hdr_top,col=SCALE_HEADER_COL,border=NA)
  for (y in c(hdr_top,hdr_bot)) segments(TX,y,TX+TW,y,lwd=1.6,lend=1,col="#111111")
  for (sx in c(TX+x_overall_sep,TX+x_bio_sep,TX+x_batch_sep,TX+x_scale_sep))
    segments(sx,hdr_bot,sx,hdr_top,lwd=0.8,col="#FFFFFF")

  hdr_cy<-(hdr_bot+hdr_top)/2
  text(TX+cum_x[1]+0.06,    hdr_cy,"Method",  cex=0.68,font=1,col="white",adj=c(0,  0.5))
  text(TX+col_cx(2),         hdr_cy,"Language",cex=0.64,font=1,col="white",adj=c(0.5,0.5))
  text(TX+col_cx(3),         hdr_cy,"Category",cex=0.64,font=1,col="white",adj=c(0.5,0.5))
  text(TX+col_cx(4),         hdr_cy,"Output",  cex=0.64,font=1,col="white",adj=c(0.5,0.5))
  text(TX+col_cx(j_overall), hdr_cy,"Overall", cex=0.60,font=1,col="white",adj=c(0.5,0.5))
  bio_mid  <- TX+x_bio_sep  +(x_batch_sep-x_bio_sep)/2
  batch_mid<- TX+x_batch_sep+(x_scale_sep-x_batch_sep)/2
  scale_mid<- TX+x_scale_sep+(TW-x_scale_sep)/2
  text(bio_mid,   hdr_cy,"Bio Conservation", cex=0.66,font=1,col="white",adj=c(0.5,0.5))
  text(batch_mid, hdr_cy,"Batch Correction",  cex=0.66,font=1,col="white",adj=c(0.5,0.5))
  text(scale_mid, hdr_cy,"Scalability",       cex=0.64,font=1,col="white",adj=c(0.5,0.5))

  all_rt  <- setNames(sapply(methods,function(m) if(m%in%rownames(perf_df))perf_df[m,"perf_total_time_s"]else NA_real_),methods)
  all_mem <- setNames(sapply(methods,function(m) if(m%in%rownames(perf_df))perf_df[m,"perf_peak_rss_gb"] else NA_real_),methods)
  rt_norm_scores  <- normalize_metric(all_rt,  higher_is_better=FALSE)
  mem_norm_scores <- normalize_metric(all_mem, higher_is_better=FALSE)
  rt_ranks  <- rank(all_rt,  ties.method="average",na.last="keep")
  mem_ranks <- rank(all_mem, ties.method="average",na.last="keep")

  draw_scale_dot <- function(j_col, norm_score, rnk, i_row) {
    bcx <- TX+col_cx(j_col); cy_row <- row_cy(i_row)
    if (is.na(norm_score)||is.na(rnk)) {
      text(bcx,cy_row,"N/A",cex=0.50,col="#BBBBBB",adj=c(0.5,0.5)); return(invisible(NULL))
    }
    r   <- MIN_R+(max(norm_score,0)^0.55)*(MAX_R-MIN_R); r<-max(MIN_R,min(MAX_R,r))
    idx <- max(1L,min(100L,as.integer(ceiling((rnk-1)/max(n_m-1,1)*99+1))))
    th  <- seq(0,2*pi,length.out=80)
    polygon(bcx+r*cos(th),cy_row+r*sin(th),col=RTPAL[idx],border="black",lwd=0.7)
  }

  for (i in seq_along(methods)) {
    m<-methods[i]; cy<-row_cy(i)
    is_empty<-all(is.na(met_df[met_df$method==m,all_avail]))
    tc<-if(is_empty)"#AAAAAA" else "#111111"; tf<-if(is_empty)3 else 1
    text(TX+cum_x[1]+0.06,cy,m,    cex=0.74,font=tf,col=tc,adj=c(0,0.5))
    text(TX+col_cx(2),cy,LANG[[m]],cex=0.62,font=1,col=if(is_empty)"#BBBBBB"else"#333333",adj=c(0.5,0.5))
    text(TX+col_cx(3),cy,CATG[[m]],cex=0.56,font=1,col=if(is_empty)"#BBBBBB"else"#333333",adj=c(0.5,0.5))
    text(TX+col_cx(4),cy,OUTPUT_TYPE[[m]],cex=0.56,font=1,col=if(is_empty)"#BBBBBB"else"#333333",adj=c(0.5,0.5))

    # Combined overall rectangle: colour=bio rank, width=batch rank
    {
      br<-bio_friedman[m]; batchr<-batch_friedman[m]
      col_left<-TX+cum_x[j_overall]; cw_ov<-col_w[j_overall]
      rect_h_ov<-ROW_H*0.62; max_w_ov<-cw_ov-2*RECT_PAD
      if (!is.na(br)&&!is.na(batchr)) {
        bio_idx  <- max(1L,min(100L,as.integer(ceiling((br-1)/max(n_m-1,1)*99+1))))
        batch_nc <- max(0,min(1,(batchr-1)/max(n_m-1,1)))
        rect_w_ov<- RECT_PAD+batch_nc*(max_w_ov-RECT_PAD)
        rect(col_left+RECT_PAD,cy-rect_h_ov/2,col_left+RECT_PAD+rect_w_ov,cy+rect_h_ov/2,
             col=OVERALL_PAL[bio_idx],border="#555555",lwd=0.6)
      } else text(TX+col_cx(j_overall),cy,"N/A",cex=0.50,col="#BBBBBB",adj=c(0.5,0.5))
    }

    # Bio dots
    for (j in seq_along(bio_cols_avail)){
      met<-bio_cols_avail[j]; jj<-j_bio_start+j-1L; bcx<-TX+col_cx(jj)
      nv<-norm_df[[met]][norm_df$method==m]
      rnk<-if(met%in%colnames(bio_mat)&&m%in%rownames(bio_mat))
             rank(-bio_mat[,met],ties.method="average",na.last="keep")[m] else NA
      if(length(nv)==0||is.na(nv)||is.na(rnk)){text(bcx,cy,"N/A",cex=0.50,col="#BBBBBB",adj=c(0.5,0.5));next}
      r<-MIN_R+(max(nv,0)^0.55)*(MAX_R-MIN_R); r<-max(MIN_R,min(MAX_R,r))
      idx<-max(1L,min(100L,as.integer(ceiling((rnk-1)/max(n_m-1,1)*99+1))))
      th<-seq(0,2*pi,length.out=80)
      polygon(bcx+r*cos(th),cy+r*sin(th),col=BIO_PAL[idx],border="black",lwd=0.7)
    }

    # Batch dots
    for (j in seq_along(batch_cols_avail)){
      met<-batch_cols_avail[j]; jj<-j_batch_start+j-1L; bcx<-TX+col_cx(jj)
      nv<-norm_df[[met]][norm_df$method==m]
      rnk<-if(met%in%colnames(batch_mat)&&m%in%rownames(batch_mat))
             rank(-batch_mat[,met],ties.method="average",na.last="keep")[m] else NA
      if(length(nv)==0||is.na(nv)||is.na(rnk)){text(bcx,cy,"N/A",cex=0.50,col="#BBBBBB",adj=c(0.5,0.5));next}
      r<-MIN_R+(max(nv,0)^0.55)*(MAX_R-MIN_R); r<-max(MIN_R,min(MAX_R,r))
      idx<-max(1L,min(100L,as.integer(ceiling((rnk-1)/max(n_m-1,1)*99+1))))
      th<-seq(0,2*pi,length.out=80)
      polygon(bcx+r*cos(th),cy+r*sin(th),col=BATCH_PAL[idx],border="black",lwd=0.7)
    }

    # Scalability dots
    draw_scale_dot(j_rt_dot,  rt_norm_scores[m], rt_ranks[m],  i)
    draw_scale_dot(j_mem_dot, mem_norm_scores[m],mem_ranks[m], i)

    # Scalability overall rect: colour=runtime, width=memory
    {
      rt_nc_bad  <- if(!is.na(rt_norm_scores[m]))  1-rt_norm_scores[m]  else NA_real_
      mem_nc_bad <- if(!is.na(mem_norm_scores[m])) 1-mem_norm_scores[m] else NA_real_
      col_left_s<-TX+cum_x[j_scale_rect]; cw_s<-col_w[j_scale_rect]
      rect_h_s<-ROW_H*0.62; max_w_s<-cw_s-2*RECT_PAD
      if (!is.na(rt_nc_bad)&&!is.na(mem_nc_bad)) {
        col_idx_s<-max(1L,min(100L,as.integer(ceiling(rt_nc_bad*99+1))))
        rect_w_s <-RECT_PAD+mem_nc_bad*(max_w_s-RECT_PAD)
        rect(col_left_s+RECT_PAD,cy-rect_h_s/2,col_left_s+RECT_PAD+rect_w_s,cy+rect_h_s/2,
             col=RTPAL[col_idx_s],border="#555555",lwd=0.6)
      } else text(TX+col_cx(j_scale_rect),cy,"N/A",cex=0.50,col="#BBBBBB",adj=c(0.5,0.5))
    }
  }

  # Rotated column labels
  text(TX+col_cx(j_overall),   tbl_bot-0.05,"Overall Bio-Batch",  cex=0.52,col="#333333",adj=c(1,0.5),srt=90)
  for (j in seq_along(bio_cols_avail)){
    jj<-j_bio_start+j-1L
    lbl<-METRIC_LABELS_FLAT[[bio_cols_avail[j]]]; if(is.null(lbl))lbl<-bio_cols_avail[j]
    text(TX+col_cx(jj),tbl_bot-0.05,lbl,cex=0.58,col="#333333",adj=c(1,0.5),srt=90)
  }
  for (j in seq_along(batch_cols_avail)){
    jj<-j_batch_start+j-1L
    lbl<-METRIC_LABELS_FLAT[[batch_cols_avail[j]]]; if(is.null(lbl))lbl<-batch_cols_avail[j]
    text(TX+col_cx(jj),tbl_bot-0.05,lbl,cex=0.58,col="#333333",adj=c(1,0.5),srt=90)
  }
  text(TX+col_cx(j_scale_rect),tbl_bot-0.05,"Overall Scalability",cex=0.58,col="#333333",adj=c(1,0.5),srt=90)
  text(TX+col_cx(j_rt_dot),    tbl_bot-0.05,"Runtime (s)",         cex=0.58,col="#333333",adj=c(1,0.5),srt=90)
  text(TX+col_cx(j_mem_dot),   tbl_bot-0.05,"Peak RAM (GB)",       cex=0.58,col="#333333",adj=c(1,0.5),srt=90)

  # Legend colour bars
  leg_x0<-TX+TW+LEG_GAP; bar_hl<-n_m*ROW_H*0.45
  bar_top<-hdr_bot-(n_m*ROW_H-bar_hl)/2; bar_y0<-bar_top-bar_hl; n_seg<-80
  draw_cbar<-function(x,y0,ytop,pal,reverse=FALSE){
    bh<-(ytop-y0)/n_seg; idx_seq<-if(reverse)rev(seq_len(n_seg))else seq_len(n_seg)
    for(s in seq_len(n_seg)){ic<-ceiling(idx_seq[s]/n_seg*100)
      rect(x,y0+(s-1)*bh,x+BAR_W_L,y0+s*bh,col=pal[max(1,min(100,ic))],border=NA)}
    rect(x,y0,x+BAR_W_L,ytop,col=NA,border="#AAAAAA",lwd=0.3)
  }
  x1<-leg_x0; draw_cbar(x1,bar_y0,bar_top,BIO_PAL,reverse=TRUE)
  text(x1+BAR_W_L/2,bar_top+0.06,"Bio\nrank",cex=0.42,col="#333333",adj=c(0.5,0))
  text(x1-0.03,bar_top,"best", cex=0.36,col="#666666",adj=c(1,0.5))
  text(x1-0.03,bar_y0, "worst",cex=0.36,col="#666666",adj=c(1,0.5))
  x2<-x1+BAR_W_L+BAR_SEP; draw_cbar(x2,bar_y0,bar_top,BATCH_PAL,reverse=TRUE)
  text(x2+BAR_W_L/2,bar_top+0.06,"Batch\nrank",cex=0.42,col="#333333",adj=c(0.5,0))
  text(x2+BAR_W_L+0.03,bar_top,"best", cex=0.36,col="#666666",adj=c(0,0.5))
  text(x2+BAR_W_L+0.03,bar_y0, "worst",cex=0.36,col="#666666",adj=c(0,0.5))
  x3<-x2+BAR_W_L+BAR_SEP; draw_cbar(x3,bar_y0,bar_top,RTPAL,reverse=TRUE)
  text(x3+BAR_W_L/2,bar_top+0.15,"Runtime /\nMemory",cex=0.42,col="#333333",adj=c(0.5,0))
  text(x3+BAR_W_L+0.03,bar_top,"low", cex=0.36,col="#666666",adj=c(0,0.5))
  text(x3+BAR_W_L+0.03,bar_y0, "high",cex=0.36,col="#666666",adj=c(0,0.5))

  sc_cx<-(x1+x2+BAR_W_L)/2+0.20; pcts<-c(0,0.25,0.50,0.75,1.00)
  sc_gap<-MAX_R*2+0.01; sc_x0<-sc_cx-(length(pcts)-1)*sc_gap/2
  sc_y<-bar_y0-MAX_R-0.22
  text(sc_cx,bar_y0-0.10,"Score (circle size)",cex=0.50,col="#333333",adj=c(0.5,1))
  for(k in seq_along(pcts)){
    r_leg<-MIN_R+(pcts[k]^0.55)*(MAX_R-MIN_R); cx_k<-sc_x0+(k-1L)*sc_gap
    th<-seq(0,2*pi,length.out=60)
    polygon(cx_k+r_leg*cos(th),sc_y+r_leg*sin(th),col="white",border="#555555",lwd=0.5)
  }
  text(sc_x0,                          sc_y-MAX_R-0.04,"0%",  cex=0.38,col="#555555",adj=c(0.5,1))
  text(sc_x0+(length(pcts)-1)*sc_gap,  sc_y-MAX_R-0.04,"100%",cex=0.38,col="#555555",adj=c(0.5,1))

  cap1<-"Overall rectangle (before bio): colour = bio Friedman rank; width = batch Friedman rank (narrow = best batch correction)."
  cap2<-"Bio/Batch dots: colour = per-metric rank; size = normalised score. Scalability dots: lighter/larger = faster or less RAM. Overall Scalability rect: colour = runtime rank (light=fast); width = memory rank (narrow=less RAM). batch_ASW & pc_regression inverted."
  text(TX,tbl_bot-LBL_H+0.08,cap1,cex=0.54,font=3,col="#555555",adj=c(0,1))
  text(TX,tbl_bot-LBL_H-0.25,cap2,cex=0.54,font=3,col="#555555",adj=c(0,1))
  message("  [saved] ",basename(out))
}

# =============================================================================
# RADAR helpers
# =============================================================================
draw_radar_labels_int<-function(n_vars,var_names,label_r=1.40,cex=0.70){
  angles<-seq(90,90-360,length.out=n_vars+1)[-(n_vars+1)]*pi/180
  for(i in seq_len(n_vars)){
    ang<-angles[i]
    ha<-if(cos(ang)>0.15)0 else if(cos(ang)<(-0.15))1 else 0.5
    va<-if(sin(ang)>0.15)0 else if(sin(ang)<(-0.15))1 else 0.5
    text(label_r*cos(ang),label_r*sin(ang),labels=var_names[i],
         cex=cex,col="#333333",adj=c(ha,va),font=1)
  }
}
draw_radar_grid<-function(n_vars,seg=4){
  angs<-seq(90,90-360,length.out=n_vars+1)[-(n_vars+1)]*pi/180
  ring_vals<-seq(1/seg,1,by=1/seg)
  for(rv in ring_vals){
    px<-rv*cos(angs); py<-rv*sin(angs)
    for(k in seq_len(n_vars)){k2<-if(k==n_vars)1L else k+1L
      lines(c(px[k],px[k2]),c(py[k],py[k2]),col="#CCCCCC",lty=2,lwd=1.0)}
  }
  for(ang in angs) lines(c(0,cos(ang)),c(0,sin(ang)),col="#CCCCCC",lwd=0.5)
  lab_ang<-angs[1]+0.12
  for(rv in ring_vals)
    text(rv*cos(lab_ang)+0.02,rv*sin(lab_ang),sprintf("%.0f%%",rv*100),
         cex=0.46,col="#AAAAAA",adj=c(0,0.5))
}

# Overlaid radar: FILLED polygons (low alpha) + lines + points
make_radar_overlaid_int<-function(metric_cols,out,title1,title2,width=10,height=11){
  methods<-intersect(method_order,rownames(norm_wide))
  nv<-length(metric_cols)
  var_lbl<-unname(sapply(metric_cols,function(x){lbl<-METRIC_LABELS[[x]];if(is.null(lbl))x else lbl}))
  angs<-seq(90,90-360,length.out=nv+1)[-(nv+1)]*pi/180

  pdf(out,width=width,height=height,family=BASE_FONT)
  on.exit(dev.off(),add=TRUE)
  par(mar=c(1,1,3.2,1),bg="white")
  plot(0,0,type="n",xlim=c(-1.75,1.75),ylim=c(-1.90,1.65),
       asp=1,axes=FALSE,xlab="",ylab="",xaxs="i",yaxs="i")
  draw_radar_grid(nv,seg=4)
  lab_ang<-angs[1]+0.14
  for(rv in c(0.25,0.50,0.75))
    text(rv*cos(lab_ang)+0.02,rv*sin(lab_ang),sprintf("%.0f%%",rv*100),
         cex=0.52,col="#AAAAAA",adj=c(0,0.5))
  draw_radar_labels_int(nv,var_lbl,label_r=1.40,cex=0.80)

  # Pass 1: filled polygons at low alpha so all methods remain visible
  for(m in methods){
    clr<-METHOD_COLORS[[m]]; if(is.null(clr)) clr<-"#555555"
    vals<-as.numeric(norm_wide[m,metric_cols]); vals[is.na(vals)]<-0
    px<-c(vals*cos(angs),vals[1]*cos(angs[1])); py<-c(vals*sin(angs),vals[1]*sin(angs[1]))
    rgb_<-col2rgb(clr)/255
    polygon(px,py,col=rgb(rgb_[1],rgb_[2],rgb_[3],alpha=0.10),border=NA)
  }
  # Pass 2: lines + points on top
  for(m in methods){
    clr<-METHOD_COLORS[[m]]; if(is.null(clr)) clr<-"#555555"
    vals<-as.numeric(norm_wide[m,metric_cols]); vals[is.na(vals)]<-0
    px<-c(vals*cos(angs),vals[1]*cos(angs[1])); py<-c(vals*sin(angs),vals[1]*sin(angs[1]))
    lines(px,py,col=clr,lwd=2.2,lend="round",ljoin="round")
    points(px[-length(px)],py[-length(py)],pch=21,cex=0.95,bg="white",col=clr,lwd=1.5)
  }

  mtext(title1,side=3,line=1.8,cex=1.05,font=2,col="#111111")
  mtext(title2,side=3,line=0.5,cex=0.72,font=3,col="#555555")
  n_leg_col<-min(4L,ceiling(length(methods)/2))
  legend(x=0,y=-1.62,xjust=0.5,yjust=0.5,xpd=TRUE,
         legend=methods,col=unname(METHOD_COLORS[methods]),pt.bg=unname(METHOD_COLORS[methods]),
         pch=21,pt.cex=1.10,lwd=2.0,lty=1,ncol=n_leg_col,
         bty="n",cex=0.78,text.col=unname(METHOD_COLORS[methods]),x.intersp=0.7,y.intersp=1.0)
  message("  [saved] ",basename(out))
}

# Small-multiples: filled polygons (alpha=0.20)
make_radar_small_multiples_int<-function(metric_cols,out,title,subtitle,
                                         width=22,height=16,ncols=4L){
  methods<-intersect(method_order,rownames(norm_wide))
  n_m<-length(methods); nv<-length(metric_cols); nrows<-ceiling(n_m/ncols)
  var_lbl<-unname(sapply(metric_cols,function(x){lbl<-METRIC_LABELS[[x]];if(is.null(lbl))x else lbl}))
  angs<-seq(90,90-360,length.out=nv+1)[-(nv+1)]*pi/180

  CairoPDF(out,width=width,height=height,family=BASE_FONT)
  par(mfrow=c(nrows,ncols),mar=c(2.2,2.2,2.8,2.2),oma=c(1.8,0,2.5,0),bg="white")
  for(m in methods){
    clr<-METHOD_COLORS[[m]]; if(is.null(clr)) clr<-"#cbc7d8"
    vals<-as.numeric(norm_wide[m,metric_cols]); vals[is.na(vals)]<-0
    px<-vals*cos(angs); py<-vals*sin(angs)
    plot(0,0,type="n",xlim=c(-1.9,1.9),ylim=c(-1.9,1.9),
         asp=1,axes=FALSE,xlab="",ylab="",xaxs="i",yaxs="i")
    draw_radar_grid(nv,seg=4)
    draw_radar_labels_int(nv,var_lbl,label_r=1.48,cex=0.64)
    rgb_<-col2rgb(clr)/255
    polygon(c(px,px[1]),c(py,py[1]),
            col=rgb(rgb_[1],rgb_[2],rgb_[3],alpha=0.20),border=NA)
    lines(c(px,px[1]),c(py,py[1]),col=clr,lwd=2.2,lend="round",ljoin="round")
    points(px,py,pch=21,cex=0.90,bg="white",col=clr,lwd=1.5)
    title(main=m,col.main=clr,font.main=2,cex.main=1.02,line=0.5)
  }
  for(b in seq_len(ncols*nrows-n_m)) plot.new()
  mtext(title,   side=3,outer=TRUE,cex=0.92,font=2,col="#111111",line=1.0)
  mtext(subtitle,side=1,outer=TRUE,cex=0.58,col="#999999",font=3, line=0.3)
  dev.off()
  message("  [saved] ",basename(out))
}

# =============================================================================
# PLAIN SCORE TABLES  (PNG, teal header)
# =============================================================================
make_plain_table<-function(metric_cols,perf_cols=NULL,out,
                            title_line1,title_line2,invert_note=NULL){
  methods<-method_order; n_m<-length(methods)
  all_cols<-c(metric_cols,perf_cols); n_data_col<-length(all_cols)

  cell<-matrix("--",nrow=n_m,ncol=n_data_col,dimnames=list(methods,all_cols))
  for(m in methods){
    for(met in metric_cols){
      rr<-met_df[met_df$method==m,]
      if(nrow(rr)>0&&met%in%names(rr)){v<-rr[[met]];if(length(v)>0&&!is.na(v))cell[m,met]<-sprintf("%.4f",round(v,4))}
    }
    if(!is.null(perf_cols)) for(pc in perf_cols){
      if(m%in%rownames(perf_df)&&pc%in%colnames(perf_df)){
        v<-perf_df[m,pc]
        if(!is.na(v)&&is.finite(v))
          cell[m,pc]<-if(pc=="perf_total_time_s")sprintf("%.1f s",v)
                      else if(pc=="perf_peak_rss_gb")sprintf("%.2f GB",v)
                      else sprintf("%.4f",v)
      }
    }
  }

  best_per<-list()
  for(met in metric_cols){
    dir<-METRIC_DIRECTION[[met]]; if(is.null(dir))dir<-TRUE
    scores<-sapply(methods,function(m){rr<-met_df[met_df$method==m,];if(nrow(rr)>0&&met%in%names(rr))rr[[met]]else NA_real_})
    best_per[[met]]<-if(dir)names(which.max(scores))else names(which.min(abs(scores)))
  }
  if(!is.null(perf_cols)) for(pc in perf_cols){
    vals<-setNames(sapply(methods,function(m)if(m%in%rownames(perf_df))perf_df[m,pc]else NA_real_),methods)
    best_per[[pc]]<-names(which.min(vals))
  }

  col_display<-sapply(all_cols,function(x){
    lbl<-METRIC_LABELS_FLAT[x][[1]]; if(!is.null(lbl))lbl else x
  },USE.NAMES=FALSE)

  W_RANK<-0.28; W_METHOD<-1.40; W_DATA<-0.96
  col_w<-c(W_RANK,W_METHOD,rep(W_DATA,n_data_col))
  cum_x<-cumsum(c(0,col_w)); TW<-sum(col_w)
  ROW_H<-0.330; HDR_H<-0.380
  L_MAR<-0.50; R_MAR<-0.40; T_MAR<-1.10; B_MAR<-0.55
  FIG_W<-L_MAR+TW+R_MAR; FIG_H<-T_MAR+HDR_H+n_m*ROW_H+B_MAR

  png(out,width=round(FIG_W*PNG_RES),height=round(FIG_H*PNG_RES),res=PNG_RES,
      bg="white",family=BASE_FONT)
  on.exit(dev.off(),add=TRUE)
  par(mar=c(0,0,0,0),bg="white")
  plot(0,0,type="n",xlim=c(0,FIG_W),ylim=c(0,FIG_H),
       asp=NA,axes=FALSE,xlab="",ylab="",xaxs="i",yaxs="i")

  TX<-L_MAR; TY<-B_MAR+HDR_H+n_m*ROW_H
  hdr_top<-TY; hdr_bot<-TY-HDR_H; data_top<-hdr_bot; tbl_bot<-data_top-n_m*ROW_H
  row_cy<-function(i) data_top-(i-0.5)*ROW_H; hdr_cy<-(hdr_top+hdr_bot)/2

  text(FIG_W/2,TY+0.72,title_line1,cex=1.00,font=2,col="#000000",adj=c(0.5,0.5))
  text(FIG_W/2,TY+0.38,title_line2,cex=0.68,font=3,col="#333333",adj=c(0.5,0.5))

  segments(TX,hdr_top,TX+TW,hdr_top,lwd=1.8,lend=1,col="#000000")
  segments(TX,hdr_bot,TX+TW,hdr_bot,lwd=0.9,lend=1,col="#000000")
  segments(TX,tbl_bot,TX+TW,tbl_bot,lwd=1.8,lend=1,col="#000000")

  for(i in seq_len(n_m)){
    shade<-if(i%%2==0)"#F0F0F0" else "white"
    rect(TX,data_top-i*ROW_H,TX+TW,data_top-(i-1)*ROW_H,col=shade,border=NA)
  }

  rect(TX,hdr_bot,TX+TW,hdr_top,col=BIO_HEADER_COL,border=NA)
  segments(TX,hdr_top,TX+TW,hdr_top,lwd=1.8,lend=1,col="#000000")
  segments(TX,hdr_bot,TX+TW,hdr_bot,lwd=0.9,lend=1,col="#000000")

  text(TX+cum_x[1]+W_RANK-0.04,hdr_cy,"#",      cex=0.76,font=2,col="white",adj=c(1,0.5))
  text(TX+cum_x[2]+0.06,        hdr_cy,"Method", cex=0.76,font=2,col="white",adj=c(0,0.5))
  for(j in seq_len(n_data_col))
    text(TX+cum_x[j+2]+W_DATA-0.06,hdr_cy,col_display[j],cex=0.64,font=2,col="white",adj=c(1,0.5))

  for(i in seq_along(methods)){
    m<-methods[i]; cy<-row_cy(i)
    is_empty<-all(is.na(met_df[met_df$method==m,all_avail]))
    tc<-if(is_empty)"#AAAAAA" else "#000000"; tf<-if(is_empty)3 else 1
    text(TX+cum_x[1]+W_RANK-0.04,cy,as.character(i),cex=0.72,font=1,col="#777777",adj=c(1,0.5))
    text(TX+cum_x[2]+0.06,cy,m,cex=0.80,font=if(i==1&&!is_empty)2 else tf,col=tc,adj=c(0,0.5))
    for(j in seq_len(n_data_col)){
      cn<-all_cols[j]; val<-cell[m,cn]; rx<-TX+cum_x[j+2]+W_DATA-0.08
      is_best<-isTRUE(best_per[[cn]]==m)
      fv<-if(is_best&&!is_empty)2 else tf
      text(rx,cy,val,cex=0.74,font=fv,col=tc,adj=c(1,0.5))
      if(is_best&&!is_empty&&val!="--"){
        tw<-nchar(val)*0.056
        segments(rx-tw*2,cy-ROW_H*0.36,rx,cy-ROW_H*0.36,lwd=0.8,col="#000000")
      }
    }
  }

  notes<-c("Bold + underline = best value in column.",
            if(!is.null(invert_note))invert_note,
            "Italics / grey = no data available.")
  text(TX,tbl_bot-0.16,paste(notes,collapse="  "),cex=0.58,font=3,col="#555555",adj=c(0,1))
  message("  [saved] ",basename(out))
}

# =============================================================================
# MAIN
# =============================================================================
all_cols_radar  <-intersect(ALL_METRICS_RADAR,  colnames(norm_wide))
bio_cols_radar  <-intersect(BIO_METRICS_RADAR,  colnames(norm_wide))
batch_cols_radar<-intersect(BATCH_METRICS_RADAR,colnames(norm_wide))

message("\n-- Bubble plot ...")
make_bubble_plot(file.path(OUTPUT_DIR,"integration_bubble_plot.pdf"))

message("\n-- Radar: all metrics overlaid (filled) ...")
make_radar_overlaid_int(all_cols_radar,
  file.path(OUTPUT_DIR,"integration_radar_all_overlaid.pdf"),
  title1="Integration method performance -- all metrics",
  title2=paste0("Normalised scores across all ",length(all_cols_radar),
                " metrics  |  higher = better  |  batch_ASW & pc_regression inverted"),
  width=10,height=11)

message("\n-- Radar: bio conservation overlaid (filled) ...")
make_radar_overlaid_int(bio_cols_radar,
  file.path(OUTPUT_DIR,"integration_radar_bio_overlaid.pdf"),
  title1="Integration method performance -- bio conservation",
  title2="Normalised bio conservation score  |  higher = better",
  width=9,height=10)

message("\n-- Radar: batch correction overlaid (filled) ...")
make_radar_overlaid_int(batch_cols_radar,
  file.path(OUTPUT_DIR,"integration_radar_batch_overlaid.pdf"),
  title1="Integration method performance -- batch correction",
  title2="Normalised batch correction score  |  higher = better  |  batch_ASW & pc_regression inverted",
  width=8,height=9)

message("\n-- Radar: all metrics small multiples (filled) ...")
make_radar_small_multiples_int(all_cols_radar,
  file.path(OUTPUT_DIR,"integration_radar_all_multiples.pdf"),
  title=paste0("Per-method performance profile  |  all ",length(all_cols_radar)," metrics  |  higher = better"),
  subtitle="All scores normalised to [0,1].  batch_ASW and pc_regression_r2_mean inverted (lower raw = better).",
  width=22,height=16,ncols=4L)

message("\n-- Radar: bio conservation small multiples (filled) ...")
make_radar_small_multiples_int(bio_cols_radar,
  file.path(OUTPUT_DIR,"integration_radar_bio_multiples.pdf"),
  title=paste0("Per-method profile  |  bio conservation (",length(bio_cols_radar)," metrics)  |  higher = better"),
  subtitle="NMI, ARI, cLISI, cell-type ASW, silhouette, isolated-label ASW/F1, rare kNN purity, Calinski-Harabasz.",
  width=18,height=12,ncols=4L)

message("\n-- Radar: batch correction small multiples (filled) ...")
make_radar_small_multiples_int(batch_cols_radar,
  file.path(OUTPUT_DIR,"integration_radar_batch_multiples.pdf"),
  title=paste0("Per-method profile  |  batch correction (",length(batch_cols_radar)," metrics)  |  higher = better"),
  subtitle="kBET, iLISI (batch), batch ASW (inverted), PC regression (inverted), graph connectivity.",
  width=16,height=10,ncols=4L)

message("\n-- Plain table: Bio conservation (PNG) ...")
make_plain_table(
  metric_cols =intersect(BIO_METRICS_RADAR,all_avail),
  out         =file.path(OUTPUT_DIR,"integration_table_bio.png"),
  title_line1 ="Table 1.  Bio conservation metrics",
  title_line2 =paste0("Raw scores for ",length(method_order)," integration methods  |  higher = better for all columns")
)

message("\n-- Plain table: Batch correction (PNG) ...")
make_plain_table(
  metric_cols =intersect(BATCH_METRICS_RADAR,all_avail),
  out         =file.path(OUTPUT_DIR,"integration_table_batch.png"),
  title_line1 ="Table 2.  Batch correction metrics",
  title_line2 =paste0("Raw scores for ",length(method_order)," integration methods"),
  invert_note ="batch_ASW: bold = min |value| (closest to 0).  pc_regression_r2_mean: bold = lowest value."
)

message("\n-- Plain table: Performance (PNG) ...")
make_plain_table(
  metric_cols =character(0),
  perf_cols   =c("perf_total_time_s","perf_peak_rss_gb"),
  out         =file.path(OUTPUT_DIR,"integration_table_performance.png"),
  title_line1 ="Table 3.  Scalability -- runtime and memory",
  title_line2 =paste0("Wall-clock runtime (s) and peak RSS memory (GB) for ",
                       length(method_order)," integration methods  |  lower = better"),
  invert_note ="Best (lowest) runtime and memory highlighted independently."
)

message("\n-- Done. All outputs in: ",OUTPUT_DIR,"\n")