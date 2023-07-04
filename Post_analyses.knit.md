
<!-- rnb-text-begin -->

---
title: "R Notebook"
output: html_notebook
---
## Setup and load data

<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxucGFjbWFuOjpwX2xvYWQocXVhbnRlZGEsIHF1YW50ZWRhLnRleHRtb2RlbHMsIHRpZHl2ZXJzZSwgcmVhZHIsIGdncGxvdDIsIHJzdGF0aXgsIGdyaWQsIGdyaWRFeHRyYSwgdmlyaWRpcywgZ2dzY2ksIGdncHViciwgcmFnZylcblxucHVsbGVkX21hbmlmZXN0b2VzIDwtIHJlYWRfY3N2KFwiZGF0YS9DTVAvcHVsbGVkX21hbmlmZXN0b2VzLmNzdlwiKVxuYGBgIn0= -->

```r
pacman::p_load(quanteda, quanteda.textmodels, tidyverse, readr, ggplot2, rstatix, grid, gridExtra, viridis, ggsci, ggpubr, ragg)

pulled_manifestoes <- read_csv("data/CMP/pulled_manifestoes.csv")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMjg2ODM5IENvbHVtbnM6IDEw4pSA4pSAIENvbHVtbiBzcGVjaWZpY2F0aW9uIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuRGVsaW1pdGVyOiBcIixcIlxuY2hyICg2KTogdGV4dCwgY29kZSwgcGFydHlfY29kZSwgY291bnRyeSwgbmFtZSwgbmFtZV9lbmdsaXNoXG5kYmwgKDQpOiBlbGVjdGlvbiwgcG9zLCBjb3VudHJ5X2NvZGUsIHBhcnR5XG7ihLkgVXNlIGBzcGVjKClgIHRvIHJldHJpZXZlIHRoZSBmdWxsIGNvbHVtbiBzcGVjaWZpY2F0aW9uIGZvciB0aGlzIGRhdGEuXG7ihLkgU3BlY2lmeSB0aGUgY29sdW1uIHR5cGVzIG9yIHNldCBgc2hvd19jb2xfdHlwZXMgPSBGQUxTRWAgdG8gcXVpZXQgdGhpcyBtZXNzYWdlLlxuIn0= -->

```
Rows: 286839 Columns: 10── Column specification ───────────────────────────────────────────
Delimiter: ","
chr (6): text, code, party_code, country, name, name_english
dbl (4): election, pos, country_code, party
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxubHJfZDJ2IDwtIHJlYWRfY3N2KFwiZGF0YS9weV9vdXRwdXRzL2xyX2Qydi5jc3ZcIiwpXG5gYGAifQ== -->

```r
lr_d2v <- read_csv("data/py_outputs/lr_d2v.csv",)
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMzggQ29sdW1uczogM+KUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIsXCJcbmNociAoMSk6IHBhcnR5X2VsZWN0aW9uXG5kYmwgKDIpOiBkMnZfZDEsIGQydl9kMlxu4oS5IFVzZSBgc3BlYygpYCB0byByZXRyaWV2ZSB0aGUgZnVsbCBjb2x1bW4gc3BlY2lmaWNhdGlvbiBmb3IgdGhpcyBkYXRhLlxu4oS5IFNwZWNpZnkgdGhlIGNvbHVtbiB0eXBlcyBvciBzZXQgYHNob3dfY29sX3R5cGVzID0gRkFMU0VgIHRvIHF1aWV0IHRoaXMgbWVzc2FnZS5cbiJ9 -->

```
Rows: 38 Columns: 3── Column specification ───────────────────────────────────────────
Delimiter: ","
chr (1): party_election
dbl (2): d2v_d1, d2v_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuZDJ2X3dmIDwtIHJlYWRfY3N2KFwiZGF0YS9weV9vdXRwdXRzL2Qydl93Zi5jc3ZcIilcbmBgYCJ9 -->

```r
d2v_wf <- read_csv("data/py_outputs/d2v_wf.csv")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMzggQ29sdW1uczogM+KUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIsXCJcbmNociAoMSk6IHBhcnR5X2VsZWN0aW9uXG5kYmwgKDIpOiBkMnZfZDEsIGQydl9kMlxu4oS5IFVzZSBgc3BlYygpYCB0byByZXRyaWV2ZSB0aGUgZnVsbCBjb2x1bW4gc3BlY2lmaWNhdGlvbiBmb3IgdGhpcyBkYXRhLlxu4oS5IFNwZWNpZnkgdGhlIGNvbHVtbiB0eXBlcyBvciBzZXQgYHNob3dfY29sX3R5cGVzID0gRkFMU0VgIHRvIHF1aWV0IHRoaXMgbWVzc2FnZS5cbiJ9 -->

```
Rows: 38 Columns: 3── Column specification ───────────────────────────────────────────
Delimiter: ","
chr (1): party_election
dbl (2): d2v_d1, d2v_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudW1hcF9zX2VtYmVkcyA8LSByZWFkX2RlbGltKFwicmVzdWx0cy9hcnJheXMvdW1hcF9zX2VtYmVkcy5jc3ZcIiwgY29sX25hbWVzPWMoXCJ1bWFwX2QxXCIsXCJ1bWFwX2QyXCIpLCBkZWxpbT1cIiBcIilcbmBgYCJ9 -->

```r
umap_s_embeds <- read_delim("results/arrays/umap_s_embeds.csv", col_names=c("umap_d1","umap_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMzggQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IHVtYXBfZDEsIHVtYXBfZDJcbuKEuSBVc2UgYHNwZWMoKWAgdG8gcmV0cmlldmUgdGhlIGZ1bGwgY29sdW1uIHNwZWNpZmljYXRpb24gZm9yIHRoaXMgZGF0YS5cbuKEuSBTcGVjaWZ5IHRoZSBjb2x1bW4gdHlwZXMgb3Igc2V0IGBzaG93X2NvbF90eXBlcyA9IEZBTFNFYCB0byBxdWlldCB0aGlzIG1lc3NhZ2UuXG4ifQ== -->

```
Rows: 38 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): umap_d1, umap_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudW1hcF93Zl9lbWJlZHMgPC0gcmVhZF9kZWxpbShcInJlc3VsdHMvYXJyYXlzL3VtYXBfd2ZfZW1iZWRzLmNzdlwiLCBjb2xfbmFtZXM9YyhcInVtYXBfZDFcIixcInVtYXBfZDJcIiksIGRlbGltPVwiIFwiKVxuYGBgIn0= -->

```r
umap_wf_embeds <- read_delim("results/arrays/umap_wf_embeds.csv", col_names=c("umap_d1","umap_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMzggQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IHVtYXBfZDEsIHVtYXBfZDJcbuKEuSBVc2UgYHNwZWMoKWAgdG8gcmV0cmlldmUgdGhlIGZ1bGwgY29sdW1uIHNwZWNpZmljYXRpb24gZm9yIHRoaXMgZGF0YS5cbuKEuSBTcGVjaWZ5IHRoZSBjb2x1bW4gdHlwZXMgb3Igc2V0IGBzaG93X2NvbF90eXBlcyA9IEZBTFNFYCB0byBxdWlldCB0aGlzIG1lc3NhZ2UuXG4ifQ== -->

```
Rows: 38 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): umap_d1, umap_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxud2ZfY21wIDwtIHJlYWRfY3N2KFwicmVzdWx0cy9hcnJheXMvd2ZfY21wLmNzdlwiKVxuYGBgIn0= -->

```r
wf_cmp <- read_csv("results/arrays/wf_cmp.csv")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMzggQ29sdW1uczogNeKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIsXCJcbmNociAoMSk6IHBhcnR5XG5kYmwgKDQpOiBlbGVjdGlvbiwgbHJfcHJvcCwgbHJfbG9nLCBscl9hYnNcbuKEuSBVc2UgYHNwZWMoKWAgdG8gcmV0cmlldmUgdGhlIGZ1bGwgY29sdW1uIHNwZWNpZmljYXRpb24gZm9yIHRoaXMgZGF0YS5cbuKEuSBTcGVjaWZ5IHRoZSBjb2x1bW4gdHlwZXMgb3Igc2V0IGBzaG93X2NvbF90eXBlcyA9IEZBTFNFYCB0byBxdWlldCB0aGlzIG1lc3NhZ2UuXG4ifQ== -->

```
Rows: 38 Columns: 5── Column specification ───────────────────────────────────────────
Delimiter: ","
chr (1): party
dbl (4): election, lr_prop, lr_log, lr_abs
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuZGZfY21wIDwtIHJlYWRfY3N2KFwicmVzdWx0cy9hcnJheXMvZGZfY21wLmNzdlwiKVxuYGBgIn0= -->

```r
df_cmp <- read_csv("results/arrays/df_cmp.csv")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMzggQ29sdW1uczogNeKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIsXCJcbmNociAoMSk6IHBhcnR5XG5kYmwgKDQpOiBlbGVjdGlvbiwgbHJfcHJvcCwgbHJfbG9nLCBscl9hYnNcbuKEuSBVc2UgYHNwZWMoKWAgdG8gcmV0cmlldmUgdGhlIGZ1bGwgY29sdW1uIHNwZWNpZmljYXRpb24gZm9yIHRoaXMgZGF0YS5cbuKEuSBTcGVjaWZ5IHRoZSBjb2x1bW4gdHlwZXMgb3Igc2V0IGBzaG93X2NvbF90eXBlcyA9IEZBTFNFYCB0byBxdWlldCB0aGlzIG1lc3NhZ2UuXG4ifQ== -->

```
Rows: 38 Columns: 5── Column specification ───────────────────────────────────────────
Delimiter: ","
chr (1): party
dbl (4): election, lr_prop, lr_log, lr_abs
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuZHJfdmFsaWQgPC0gcmVhZF9jc3YoXCJyZXN1bHRzL2FycmF5cy9kcl92YWxpZF9zY29yZXMuY3N2XCIpXG5gYGAifQ== -->

```r
dr_valid <- read_csv("results/arrays/dr_valid_scores.csv")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogNiBDb2x1bW5zOiAz4pSA4pSAIENvbHVtbiBzcGVjaWZpY2F0aW9uIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuRGVsaW1pdGVyOiBcIixcIlxuY2hyICgxKTogdGVjaG5pcXVlc1xuZGJsICgyKTogdHJ1c3R3b3J0aGluZXNzLCBzaWxob3VldHRlXG7ihLkgVXNlIGBzcGVjKClgIHRvIHJldHJpZXZlIHRoZSBmdWxsIGNvbHVtbiBzcGVjaWZpY2F0aW9uIGZvciB0aGlzIGRhdGEuXG7ihLkgU3BlY2lmeSB0aGUgY29sdW1uIHR5cGVzIG9yIHNldCBgc2hvd19jb2xfdHlwZXMgPSBGQUxTRWAgdG8gcXVpZXQgdGhpcyBtZXNzYWdlLlxuIn0= -->

```
Rows: 6 Columns: 3── Column specification ───────────────────────────────────────────
Delimiter: ","
chr (1): techniques
dbl (2): trustworthiness, silhouette
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxucGNhX3NjYWxlZCA8LSByZWFkX2RlbGltKFwicmVzdWx0cy9hcnJheXMvcGNhX3NjYWxlZC5jc3ZcIiwgY29sX25hbWVzPWMoXCJwY2FfZDFcIixcInBjYV9kMlwiKSwgZGVsaW09XCIgXCIpXG5gYGAifQ== -->

```r
pca_scaled <- read_delim("results/arrays/pca_scaled.csv", col_names=c("pca_d1","pca_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMjYwOTYgQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IHBjYV9kMSwgcGNhX2QyXG7ihLkgVXNlIGBzcGVjKClgIHRvIHJldHJpZXZlIHRoZSBmdWxsIGNvbHVtbiBzcGVjaWZpY2F0aW9uIGZvciB0aGlzIGRhdGEuXG7ihLkgU3BlY2lmeSB0aGUgY29sdW1uIHR5cGVzIG9yIHNldCBgc2hvd19jb2xfdHlwZXMgPSBGQUxTRWAgdG8gcXVpZXQgdGhpcyBtZXNzYWdlLlxuIn0= -->

```
Rows: 26096 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): pca_d1, pca_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudW1hcF9zX3NjYWxlZCA8LSByZWFkX2RlbGltKFwicmVzdWx0cy9hcnJheXMvdW1hcF9zX3NjYWxlZC5jc3ZcIiwgY29sX25hbWVzPWMoXCJ1bWFwX3NfZDFcIixcInVtYXBfc19kMlwiKSwgZGVsaW09XCIgXCIpXG5gYGAifQ== -->

```r
umap_s_scaled <- read_delim("results/arrays/umap_s_scaled.csv", col_names=c("umap_s_d1","umap_s_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMjYwOTYgQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IHVtYXBfc19kMSwgdW1hcF9zX2QyXG7ihLkgVXNlIGBzcGVjKClgIHRvIHJldHJpZXZlIHRoZSBmdWxsIGNvbHVtbiBzcGVjaWZpY2F0aW9uIGZvciB0aGlzIGRhdGEuXG7ihLkgU3BlY2lmeSB0aGUgY29sdW1uIHR5cGVzIG9yIHNldCBgc2hvd19jb2xfdHlwZXMgPSBGQUxTRWAgdG8gcXVpZXQgdGhpcyBtZXNzYWdlLlxuIn0= -->

```
Rows: 26096 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): umap_s_d1, umap_s_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudW1hcF91c19zY2FsZWQgPC0gcmVhZF9kZWxpbShcInJlc3VsdHMvYXJyYXlzL3VtYXBfdXNfc2NhbGVkLmNzdlwiLCBjb2xfbmFtZXM9YyhcInVtYXBfdXNfZDFcIixcInVtYXBfdXNfZDJcIiksIGRlbGltPVwiIFwiKVxuYGBgIn0= -->

```r
umap_us_scaled <- read_delim("results/arrays/umap_us_scaled.csv", col_names=c("umap_us_d1","umap_us_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMjYwOTYgQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IHVtYXBfdXNfZDEsIHVtYXBfdXNfZDJcbuKEuSBVc2UgYHNwZWMoKWAgdG8gcmV0cmlldmUgdGhlIGZ1bGwgY29sdW1uIHNwZWNpZmljYXRpb24gZm9yIHRoaXMgZGF0YS5cbuKEuSBTcGVjaWZ5IHRoZSBjb2x1bW4gdHlwZXMgb3Igc2V0IGBzaG93X2NvbF90eXBlcyA9IEZBTFNFYCB0byBxdWlldCB0aGlzIG1lc3NhZ2UuXG4ifQ== -->

```
Rows: 26096 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): umap_us_d1, umap_us_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuYWVfc2NhbGVkIDwtIHJlYWRfZGVsaW0oXCJyZXN1bHRzL2FycmF5cy9hZV9zY2FsZWQuY3N2XCIsY29sX25hbWVzPWMoXCJhZV9kMVwiLFwiYWVfZDJcIiksIGRlbGltPVwiIFwiKVxuYGBgIn0= -->

```r
ae_scaled <- read_delim("results/arrays/ae_scaled.csv",col_names=c("ae_d1","ae_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMjYwOTYgQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IGFlX2QxLCBhZV9kMlxu4oS5IFVzZSBgc3BlYygpYCB0byByZXRyaWV2ZSB0aGUgZnVsbCBjb2x1bW4gc3BlY2lmaWNhdGlvbiBmb3IgdGhpcyBkYXRhLlxu4oS5IFNwZWNpZnkgdGhlIGNvbHVtbiB0eXBlcyBvciBzZXQgYHNob3dfY29sX3R5cGVzID0gRkFMU0VgIHRvIHF1aWV0IHRoaXMgbWVzc2FnZS5cbiJ9 -->

```
Rows: 26096 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): ae_d1, ae_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuaXZpc19zY2FsZWQgPC0gcmVhZF9kZWxpbShcInJlc3VsdHMvYXJyYXlzL2l2aXNfc2NhbGVkLmNzdlwiLGNvbF9uYW1lcz1jKFwiaXZpc19kMVwiLFwiaXZpc19kMlwiKSwgZGVsaW09XCIgXCIpXG5gYGAifQ== -->

```r
ivis_scaled <- read_delim("results/arrays/ivis_scaled.csv",col_names=c("ivis_d1","ivis_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMjYwOTYgQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IGl2aXNfZDEsIGl2aXNfZDJcbuKEuSBVc2UgYHNwZWMoKWAgdG8gcmV0cmlldmUgdGhlIGZ1bGwgY29sdW1uIHNwZWNpZmljYXRpb24gZm9yIHRoaXMgZGF0YS5cbuKEuSBTcGVjaWZ5IHRoZSBjb2x1bW4gdHlwZXMgb3Igc2V0IGBzaG93X2NvbF90eXBlcyA9IEZBTFNFYCB0byBxdWlldCB0aGlzIG1lc3NhZ2UuXG4ifQ== -->

```
Rows: 26096 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): ivis_d1, ivis_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxubGRhX3NjYWxlZCA8LSByZWFkX2RlbGltKFwicmVzdWx0cy9hcnJheXMvbGRhX3NjYWxlZC5jc3ZcIiwgY29sX25hbWVzPWMoXCJsZGFfZDFcIixcImxkYV9kMlwiKSwgZGVsaW09XCIgXCIpXG5gYGAifQ== -->

```r
lda_scaled <- read_delim("results/arrays/lda_scaled.csv", col_names=c("lda_d1","lda_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMjYwOTYgQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IGxkYV9kMSwgbGRhX2QyXG7ihLkgVXNlIGBzcGVjKClgIHRvIHJldHJpZXZlIHRoZSBmdWxsIGNvbHVtbiBzcGVjaWZpY2F0aW9uIGZvciB0aGlzIGRhdGEuXG7ihLkgU3BlY2lmeSB0aGUgY29sdW1uIHR5cGVzIG9yIHNldCBgc2hvd19jb2xfdHlwZXMgPSBGQUxTRWAgdG8gcXVpZXQgdGhpcyBtZXNzYWdlLlxuIn0= -->

```
Rows: 26096 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): lda_d1, lda_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxucGNhX2VtYmVkcyA8LSByZWFkX2RlbGltKFwicmVzdWx0cy9hcnJheXMvcGNhX2VtYmVkcy5jc3ZcIiwgY29sX25hbWVzPWMoXCJwY2FfZDFcIixcInBjYV9kMlwiKSwgZGVsaW09XCIgXCIpXG5gYGAifQ== -->

```r
pca_embeds <- read_delim("results/arrays/pca_embeds.csv", col_names=c("pca_d1","pca_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMzggQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IHBjYV9kMSwgcGNhX2QyXG7ihLkgVXNlIGBzcGVjKClgIHRvIHJldHJpZXZlIHRoZSBmdWxsIGNvbHVtbiBzcGVjaWZpY2F0aW9uIGZvciB0aGlzIGRhdGEuXG7ihLkgU3BlY2lmeSB0aGUgY29sdW1uIHR5cGVzIG9yIHNldCBgc2hvd19jb2xfdHlwZXMgPSBGQUxTRWAgdG8gcXVpZXQgdGhpcyBtZXNzYWdlLlxuIn0= -->

```
Rows: 38 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): pca_d1, pca_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudW1hcF9zX2VtYmVkcyA8LSByZWFkX2RlbGltKFwicmVzdWx0cy9hcnJheXMvdW1hcF9zX2VtYmVkcy5jc3ZcIiwgY29sX25hbWVzPWMoXCJ1bWFwX3NfZDFcIixcInVtYXBfc19kMlwiKSwgZGVsaW09XCIgXCIpXG5gYGAifQ== -->

```r
umap_s_embeds <- read_delim("results/arrays/umap_s_embeds.csv", col_names=c("umap_s_d1","umap_s_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMzggQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IHVtYXBfc19kMSwgdW1hcF9zX2QyXG7ihLkgVXNlIGBzcGVjKClgIHRvIHJldHJpZXZlIHRoZSBmdWxsIGNvbHVtbiBzcGVjaWZpY2F0aW9uIGZvciB0aGlzIGRhdGEuXG7ihLkgU3BlY2lmeSB0aGUgY29sdW1uIHR5cGVzIG9yIHNldCBgc2hvd19jb2xfdHlwZXMgPSBGQUxTRWAgdG8gcXVpZXQgdGhpcyBtZXNzYWdlLlxuIn0= -->

```
Rows: 38 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): umap_s_d1, umap_s_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudW1hcF91c19lbWJlZHMgPC0gcmVhZF9kZWxpbShcInJlc3VsdHMvYXJyYXlzL3VtYXBfdXNfZW1iZWRzLmNzdlwiLCBjb2xfbmFtZXM9YyhcInVtYXBfdXNfZDFcIixcInVtYXBfdXNfZDJcIiksIGRlbGltPVwiIFwiKVxuYGBgIn0= -->

```r
umap_us_embeds <- read_delim("results/arrays/umap_us_embeds.csv", col_names=c("umap_us_d1","umap_us_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMzggQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IHVtYXBfdXNfZDEsIHVtYXBfdXNfZDJcbuKEuSBVc2UgYHNwZWMoKWAgdG8gcmV0cmlldmUgdGhlIGZ1bGwgY29sdW1uIHNwZWNpZmljYXRpb24gZm9yIHRoaXMgZGF0YS5cbuKEuSBTcGVjaWZ5IHRoZSBjb2x1bW4gdHlwZXMgb3Igc2V0IGBzaG93X2NvbF90eXBlcyA9IEZBTFNFYCB0byBxdWlldCB0aGlzIG1lc3NhZ2UuXG4ifQ== -->

```
Rows: 38 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): umap_us_d1, umap_us_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuYWVfZW1iZWRzIDwtIHJlYWRfZGVsaW0oXCJyZXN1bHRzL2FycmF5cy9hZV9lbWJlZHMuY3N2XCIsY29sX25hbWVzPWMoXCJhZV9kMVwiLFwiYWVfZDJcIiksIGRlbGltPVwiIFwiKVxuYGBgIn0= -->

```r
ae_embeds <- read_delim("results/arrays/ae_embeds.csv",col_names=c("ae_d1","ae_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMzggQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IGFlX2QxLCBhZV9kMlxu4oS5IFVzZSBgc3BlYygpYCB0byByZXRyaWV2ZSB0aGUgZnVsbCBjb2x1bW4gc3BlY2lmaWNhdGlvbiBmb3IgdGhpcyBkYXRhLlxu4oS5IFNwZWNpZnkgdGhlIGNvbHVtbiB0eXBlcyBvciBzZXQgYHNob3dfY29sX3R5cGVzID0gRkFMU0VgIHRvIHF1aWV0IHRoaXMgbWVzc2FnZS5cbiJ9 -->

```
Rows: 38 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): ae_d1, ae_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuaXZpc19lbWJlZHMgPC0gcmVhZF9kZWxpbShcInJlc3VsdHMvYXJyYXlzL2l2aXNfZW1iZWRzLmNzdlwiLGNvbF9uYW1lcz1jKFwiaXZpc19kMVwiLFwiaXZpc19kMlwiKSwgZGVsaW09XCIgXCIpXG5gYGAifQ== -->

```r
ivis_embeds <- read_delim("results/arrays/ivis_embeds.csv",col_names=c("ivis_d1","ivis_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMzggQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IGl2aXNfZDEsIGl2aXNfZDJcbuKEuSBVc2UgYHNwZWMoKWAgdG8gcmV0cmlldmUgdGhlIGZ1bGwgY29sdW1uIHNwZWNpZmljYXRpb24gZm9yIHRoaXMgZGF0YS5cbuKEuSBTcGVjaWZ5IHRoZSBjb2x1bW4gdHlwZXMgb3Igc2V0IGBzaG93X2NvbF90eXBlcyA9IEZBTFNFYCB0byBxdWlldCB0aGlzIG1lc3NhZ2UuXG4ifQ== -->

```
Rows: 38 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): ivis_d1, ivis_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxubGRhX2VtYmVkcyA8LSByZWFkX2RlbGltKFwicmVzdWx0cy9hcnJheXMvbGRhX2VtYmVkcy5jc3ZcIiwgY29sX25hbWVzPWMoXCJsZGFfZDFcIixcImxkYV9kMlwiKSwgZGVsaW09XCIgXCIpXG5gYGAifQ== -->

```r
lda_embeds <- read_delim("results/arrays/lda_embeds.csv", col_names=c("lda_d1","lda_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogMzggQ29sdW1uczogMuKUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIgXCJcbmRibCAoMik6IGxkYV9kMSwgbGRhX2QyXG7ihLkgVXNlIGBzcGVjKClgIHRvIHJldHJpZXZlIHRoZSBmdWxsIGNvbHVtbiBzcGVjaWZpY2F0aW9uIGZvciB0aGlzIGRhdGEuXG7ihLkgU3BlY2lmeSB0aGUgY29sdW1uIHR5cGVzIG9yIHNldCBgc2hvd19jb2xfdHlwZXMgPSBGQUxTRWAgdG8gcXVpZXQgdGhpcyBtZXNzYWdlLlxuIn0= -->

```
Rows: 38 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): lda_d1, lda_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudHJ1bXBfZW1iZWRzIDwtIHJlYWRfZGVsaW0oXCJyZXN1bHRzL2FycmF5cy90cnVtcF9lbWJlZHMuY3N2XCIsY29sX25hbWVzPWMoXCJlbWJfZDFcIixcImVtYl9kMlwiKSwgZGVsaW09XCIgXCIpXG5gYGAifQ== -->

```r
trump_embeds <- read_delim("results/arrays/trump_embeds.csv",col_names=c("emb_d1","emb_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogNiBDb2x1bW5zOiAy4pSA4pSAIENvbHVtbiBzcGVjaWZpY2F0aW9uIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuRGVsaW1pdGVyOiBcIiBcIlxuZGJsICgyKTogZW1iX2QxLCBlbWJfZDJcbuKEuSBVc2UgYHNwZWMoKWAgdG8gcmV0cmlldmUgdGhlIGZ1bGwgY29sdW1uIHNwZWNpZmljYXRpb24gZm9yIHRoaXMgZGF0YS5cbuKEuSBTcGVjaWZ5IHRoZSBjb2x1bW4gdHlwZXMgb3Igc2V0IGBzaG93X2NvbF90eXBlcyA9IEZBTFNFYCB0byBxdWlldCB0aGlzIG1lc3NhZ2UuXG4ifQ== -->

```
Rows: 6 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): emb_d1, emb_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudHJ1bXBfc2NhbGVkIDwtIHJlYWRfZGVsaW0oXCJyZXN1bHRzL2FycmF5cy90cnVtcF9zY2FsZWQuY3N2XCIsY29sX25hbWVzPWMoXCJlbWJfZDFcIixcImVtYl9kMlwiKSwgZGVsaW09XCIgXCIpXG5gYGAifQ== -->

```r
trump_scaled <- read_delim("results/arrays/trump_scaled.csv",col_names=c("emb_d1","emb_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogNzE0NiBDb2x1bW5zOiAy4pSA4pSAIENvbHVtbiBzcGVjaWZpY2F0aW9uIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuRGVsaW1pdGVyOiBcIiBcIlxuZGJsICgyKTogZW1iX2QxLCBlbWJfZDJcbuKEuSBVc2UgYHNwZWMoKWAgdG8gcmV0cmlldmUgdGhlIGZ1bGwgY29sdW1uIHNwZWNpZmljYXRpb24gZm9yIHRoaXMgZGF0YS5cbuKEuSBTcGVjaWZ5IHRoZSBjb2x1bW4gdHlwZXMgb3Igc2V0IGBzaG93X2NvbF90eXBlcyA9IEZBTFNFYCB0byBxdWlldCB0aGlzIG1lc3NhZ2UuXG4ifQ== -->

```
Rows: 7146 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): emb_d1, emb_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudHJ1bXBfZW1iZWRzXzMwIDwtIHJlYWRfZGVsaW0oXCJyZXN1bHRzL2FycmF5cy90cnVtcF9lbWJlZHNfMzAuY3N2XCIsY29sX25hbWVzPWMoXCJlbWJfMzBfZDFcIixcImVtYl8zMF9kMlwiKSwgZGVsaW09XCIgXCIpXG5gYGAifQ== -->

```r
trump_embeds_30 <- read_delim("results/arrays/trump_embeds_30.csv",col_names=c("emb_30_d1","emb_30_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogNiBDb2x1bW5zOiAy4pSA4pSAIENvbHVtbiBzcGVjaWZpY2F0aW9uIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuRGVsaW1pdGVyOiBcIiBcIlxuZGJsICgyKTogZW1iXzMwX2QxLCBlbWJfMzBfZDJcbuKEuSBVc2UgYHNwZWMoKWAgdG8gcmV0cmlldmUgdGhlIGZ1bGwgY29sdW1uIHNwZWNpZmljYXRpb24gZm9yIHRoaXMgZGF0YS5cbuKEuSBTcGVjaWZ5IHRoZSBjb2x1bW4gdHlwZXMgb3Igc2V0IGBzaG93X2NvbF90eXBlcyA9IEZBTFNFYCB0byBxdWlldCB0aGlzIG1lc3NhZ2UuXG4ifQ== -->

```
Rows: 6 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): emb_30_d1, emb_30_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudHJ1bXBfc2NhbGVkXzMwIDwtIHJlYWRfZGVsaW0oXCJyZXN1bHRzL2FycmF5cy90cnVtcF9zY2FsZWRfMzAuY3N2XCIsY29sX25hbWVzPWMoXCJlbWJfMzBfZDFcIixcImVtYl8zMF9kMlwiKSwgZGVsaW09XCIgXCIpXG5gYGAifQ== -->

```r
trump_scaled_30 <- read_delim("results/arrays/trump_scaled_30.csv",col_names=c("emb_30_d1","emb_30_d2"), delim=" ")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogNzE0NiBDb2x1bW5zOiAy4pSA4pSAIENvbHVtbiBzcGVjaWZpY2F0aW9uIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuRGVsaW1pdGVyOiBcIiBcIlxuZGJsICgyKTogZW1iXzMwX2QxLCBlbWJfMzBfZDJcbuKEuSBVc2UgYHNwZWMoKWAgdG8gcmV0cmlldmUgdGhlIGZ1bGwgY29sdW1uIHNwZWNpZmljYXRpb24gZm9yIHRoaXMgZGF0YS5cbuKEuSBTcGVjaWZ5IHRoZSBjb2x1bW4gdHlwZXMgb3Igc2V0IGBzaG93X2NvbF90eXBlcyA9IEZBTFNFYCB0byBxdWlldCB0aGlzIG1lc3NhZ2UuXG4ifQ== -->

```
Rows: 7146 Columns: 2── Column specification ───────────────────────────────────────────
Delimiter: " "
dbl (2): emb_30_d1, emb_30_d2
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxudHJ1bXBfdHdlZXRzIDwtIHJlYWRfY3N2KFwiZGF0YS9zaW5zX3R3ZWV0cy9NT1ROX3Jlc3BvbnNlc19ncm91bmR0cnV0aC5jc3ZcIilcbmBgYCJ9 -->

```r
trump_tweets <- read_csv("data/sins_tweets/MOTN_responses_groundtruth.csv")
```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUm93czogNzE0NiBDb2x1bW5zOiAxM+KUgOKUgCBDb2x1bW4gc3BlY2lmaWNhdGlvbiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbkRlbGltaXRlcjogXCIsXCJcbmNociAgKDIpOiBpZGVvNSwgZWRpdHNfY2xlYW5fdGV4dFxuZGJsICgxMSk6IHdhdmVudW0sIHFwb3MsIHRydW1wX3N0YW5jZV9hdXRvLCBsZXhpY29kZXJfc2VudGltZW50Li4uXG7ihLkgVXNlIGBzcGVjKClgIHRvIHJldHJpZXZlIHRoZSBmdWxsIGNvbHVtbiBzcGVjaWZpY2F0aW9uIGZvciB0aGlzIGRhdGEuXG7ihLkgU3BlY2lmeSB0aGUgY29sdW1uIHR5cGVzIG9yIHNldCBgc2hvd19jb2xfdHlwZXMgPSBGQUxTRWAgdG8gcXVpZXQgdGhpcyBtZXNzYWdlLlxuIn0= -->

```
Rows: 7146 Columns: 13── Column specification ───────────────────────────────────────────
Delimiter: ","
chr  (2): ideo5, edits_clean_text
dbl (11): wavenum, qpos, trump_stance_auto, lexicoder_sentiment...
ℹ Use `spec()` to retrieve the full column specification for this data.
ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->


## Wordfish scaling

<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuZGZtYXQgPC0gY29ycHVzKGdyb3VwZWRfdGV4dHMsIHRleHRfZmllbGQgPSBcInRleHRfbmV3XCIpICU+JVxuICAgICAgICAgICAgIHRva2VucyhyZW1vdmVfcHVuY3QgPSBUUlVFLCByZW1vdmVfbnVtYmVycyA9IFRSVUUsIHJlbW92ZV9zeW1ib2xzID0gVFJVRSkgJT4lXG4gICAgICAgICAgICAgdG9rZW5zX3JlbW92ZShzdG9wd29yZHMoXCJkZVwiKSkgJT4lXG4gICAgICAgICAgICAgZGZtKCkgJT4lXG4gICAgICAgICAgICAgZGZtX3RyaW0obWluX3Rlcm1mcmVxID0gNTAsIHZlcmJvc2UgPSBUUlVFLCBtaW5fZG9jZnJlcSA9IDAuMDMsIGRvY2ZyZXFfdHlwZSA9IFwicHJvcFwiKSAlPiVcbiAgICAgICAgICAgICBkZm1fc3Vic2V0KG50b2tlbiguKSA+IDAsIGRyb3BfZG9jaWQgPSBUUlVFKVxuXG5gYGAifQ== -->

```r
dfmat <- corpus(grouped_texts, text_field = "text_new") %>%
             tokens(remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE) %>%
             tokens_remove(stopwords("de")) %>%
             dfm() %>%
             dfm_trim(min_termfreq = 50, verbose = TRUE, min_docfreq = 0.03, docfreq_type = "prop") %>%
             dfm_subset(ntoken(.) > 0, drop_docid = TRUE)

```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUmVtb3ZpbmcgZmVhdHVyZXMgb2NjdXJyaW5nOiBcbiAgLSBmZXdlciB0aGFuIDUwIHRpbWVzOiA2MSw2OTJcbiAgLSBpbiBmZXdlciB0aGFuIDEuMTQgZG9jdW1lbnRzOiAzNSwwNDhcbiAgVG90YWwgZmVhdHVyZXMgcmVtb3ZlZDogNjEsNjkzICg5Ni44JSkuXG4ifQ== -->

```
Removing features occurring: 
  - fewer than 50 times: 61,692
  - in fewer than 1.14 documents: 35,048
  Total features removed: 61,693 (96.8%).
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



## Merge datasets


<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxubHJfZDJ2IDwtIGxyX2QydiAlPiUgc2VwYXJhdGUoXCJwYXJ0eV9lbGVjdGlvblwiLCBzZXA9XCJcXFxcX1wiLCBpbnRvID0gYyhcInBhcnR5XCIsIFwiZWxlY3Rpb25cIikpICU+JVxuICAgIG11dGF0ZShlbGVjdGlvbiA9IGFzX251bWVyaWMoZWxlY3Rpb24pKVxuXG5gYGAifQ== -->

```r
lr_d2v <- lr_d2v %>% separate("party_election", sep="\\_", into = c("party", "election")) %>%
    mutate(election = as_numeric(election))

```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiRXJyb3IgaW4gYG11dGF0ZSgpYDpcbuKEuSBJbiBhcmd1bWVudDogYGVsZWN0aW9uID0gYXNfbnVtZXJpYyhlbGVjdGlvbilgLlxuQ2F1c2VkIGJ5IGVycm9yIGluIGBhc19udW1lcmljKClgOlxuISBjb3VsZCBub3QgZmluZCBmdW5jdGlvbiBcImFzX251bWVyaWNcIlxuQmFja3RyYWNlOlxuIDEuIC4uLiAlPiUgbXV0YXRlKGVsZWN0aW9uID0gYXNfbnVtZXJpYyhlbGVjdGlvbikpXG4gMy4gZHBseXI6OjptdXRhdGUuZGF0YS5mcmFtZSguLCBlbGVjdGlvbiA9IGFzX251bWVyaWMoZWxlY3Rpb24pKVxuIDQuIGRwbHlyOjo6bXV0YXRlX2NvbHMoLmRhdGEsIGRwbHlyX3F1b3N1cmVzKC4uLiksIGJ5KVxuIDYuIGRwbHlyOjo6bXV0YXRlX2NvbChkb3RzW1tpXV0sIGRhdGEsIG1hc2ssIG5ld19jb2x1bW5zKVxuIDcuIG1hc2skZXZhbF9hbGxfbXV0YXRlKHF1bylcbiA4LiBkcGx5ciAobG9jYWwpIGV2YWwoKVxuIn0= -->

```
Error in `mutate()`:
ℹ In argument: `election = as_numeric(election)`.
Caused by error in `as_numeric()`:
! could not find function "as_numeric"
Backtrace:
 1. ... %>% mutate(election = as_numeric(election))
 3. dplyr:::mutate.data.frame(., election = as_numeric(election))
 4. dplyr:::mutate_cols(.data, dplyr_quosures(...), by)
 6. dplyr:::mutate_col(dots[[i]], data, mask, new_columns)
 7. mask$eval_all_mutate(quo)
 8. dplyr (local) eval()
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->





## Figure 2

<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxucGxvdF93ZiA8LSBnZ3Bsb3QoZGF0YSA9IGRmLFxuICAgICAgICAgICAgICAgICAgYWVzKHg9ZmFjdG9yKGVsZWN0aW9uKSwgeT13Zl9zY29yZSwgY29sb3I9cGFydHksZ3JvdXA9cGFydHkpKSArXG4gICAgZ2VvbV9wb2ludCgpICtcbiAgICBnZW9tX2xpbmUoYWxwaGE9MC41LCBsaW5ld2lkdGggPSAxKSArIFxuICAgIHNjYWxlX2NvbG9yX21hbnVhbChuYW1lID0gXCJQYXJ0eVwiLCAgXG4gICAgICAgICAgICAgICAgICAgICAgIHZhbHVlcyA9IGMoXCJibHVlXCIsXCJibGFja1wiLCBcImdyZWVuXCIsXCJ5ZWxsb3dcIiwgXCJwdXJwbGVcIiwgXCJyZWRcIikpICtcbiAgICB0aGVtZShheGlzLnRpdGxlID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNSksbGVnZW5kLnRpdGxlID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNSksIGxlZ2VuZC50ZXh0ID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNCksIHRpdGxlID0gIGVsZW1lbnRfdGV4dChzaXplID0gMTgsIGZhY2UgPSBcImJvbGRcIikpICtcbiAgICB4bGFiKFwiRWxlY3Rpb25cIikgK1xuICAgIHlsYWIoXCJXb3JkZmlzaFwiKVxuXG5cbnBsb3RfZDJ2MSA8LSBnZ3Bsb3QoZGF0YSA9IGRmLFxuICAgICAgICAgICAgICAgICAgYWVzKHg9ZmFjdG9yKGVsZWN0aW9uKSwgeT0gZDJ2X2QxLCBjb2xvcj1wYXJ0eSxncm91cD1wYXJ0eSkpICtcbiAgICBnZW9tX3BvaW50KCkgK1xuICAgIGdlb21fbGluZShhbHBoYT0wLjUsIHNpemUgPSAxKSArIFxuICAgIHNjYWxlX2NvbG9yX21hbnVhbChuYW1lID0gXCJQYXJ0eVwiLCAgXG4gICAgICAgICAgICAgICAgICAgICAgIHZhbHVlcyA9IGMoXCJibHVlXCIsXCJibGFja1wiLCBcImdyZWVuXCIsXCJ5ZWxsb3dcIiwgXCJwdXJwbGVcIiwgXCJyZWRcIikpICtcbiAgICB0aGVtZShheGlzLnRpdGxlID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNSksbGVnZW5kLnRpdGxlID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNSksIGxlZ2VuZC50ZXh0ID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNCksIHRpdGxlID0gIGVsZW1lbnRfdGV4dChzaXplID0gMTgsIGZhY2UgPSBcImJvbGRcIikpICtcbiAgICB4bGFiKFwiRWxlY3Rpb25cIikgK1xuICAgIHlsYWIoXCJEb2MyVmVjIC0gRGltIDFcIilcblxucGxvdF9kMnYyIDwtIGdncGxvdChkYXRhID0gZGYsXG4gICAgICAgICAgICAgICAgICBhZXMoeD1mYWN0b3IoZWxlY3Rpb24pLCB5PSBkMnZfZDIsIGNvbG9yPXBhcnR5LGdyb3VwPXBhcnR5KSkgK1xuICAgIGdlb21fcG9pbnQoKSArXG4gICAgZ2VvbV9saW5lKGFscGhhPTAuNSwgc2l6ZSA9IDEpICsgXG4gICAgc2NhbGVfY29sb3JfbWFudWFsKG5hbWUgPSBcIlBhcnR5XCIsICBcbiAgICAgICAgICAgICAgICAgICAgICAgdmFsdWVzID0gYyhcImJsdWVcIixcImJsYWNrXCIsIFwiZ3JlZW5cIixcInllbGxvd1wiLCBcInB1cnBsZVwiLCBcInJlZFwiKSkgK1xuICAgIHRoZW1lKGF4aXMudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSxsZWdlbmQudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSwgbGVnZW5kLnRleHQgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE0KSwgdGl0bGUgPSAgZWxlbWVudF90ZXh0KHNpemUgPSAxOCwgZmFjZSA9IFwiYm9sZFwiKSkgK1xuICAgIHhsYWIoXCJFbGVjdGlvblwiKSArXG4gICAgeWxhYihcIkRvYzJWZWMgLSBEaW0gMlwiKVxuXG5cbnBsb3RfdGYxIDwtIGdncGxvdChkYXRhID0gZGYsXG4gICAgICAgICAgICAgICAgICBhZXMoeD1mYWN0b3IoZWxlY3Rpb24pLCB5PXVtYXBfZDEsIGNvbG9yPXBhcnR5LGdyb3VwPXBhcnR5KSkgK1xuICAgIGdlb21fcG9pbnQoKSArXG4gICAgZ2VvbV9saW5lKGFscGhhPTAuNSwgc2l6ZSA9IDEpICsgXG4gICAgc2NhbGVfY29sb3JfbWFudWFsKG5hbWUgPSBcIlBhcnR5XCIsICBcbiAgICAgICAgICAgICAgICAgICAgICAgdmFsdWVzID0gYyhcImJsdWVcIixcImJsYWNrXCIsIFwiZ3JlZW5cIixcInllbGxvd1wiLCBcInB1cnBsZVwiLCBcInJlZFwiKSkgK1xuICAgIHRoZW1lKGF4aXMudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSxsZWdlbmQudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSwgbGVnZW5kLnRleHQgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE0KSwgdGl0bGUgPSAgZWxlbWVudF90ZXh0KHNpemUgPSAxOCwgZmFjZSA9IFwiYm9sZFwiKSkgK1xuICAgIHhsYWIoXCJFbGVjdGlvblwiKSArXG4gICAgeWxhYihcIlRGU2NhbGUgLSBEaW0gMVwiKVxuXG5cbnBsb3RfdGYyIDwtIGdncGxvdChkYXRhID0gZGYsXG4gICAgICAgICAgICAgICAgICBhZXMoeD1mYWN0b3IoZWxlY3Rpb24pLCB5PXVtYXBfZDIsIGNvbG9yPXBhcnR5LGdyb3VwPXBhcnR5KSkgK1xuICAgIGdlb21fcG9pbnQoKSArXG4gICAgZ2VvbV9saW5lKGFscGhhPTAuNSwgc2l6ZSA9IDEpICsgXG4gICAgc2NhbGVfY29sb3JfbWFudWFsKG5hbWUgPSBcIlBhcnR5XCIsICBcbiAgICAgICAgICAgICAgICAgICAgICAgdmFsdWVzID0gYyhcImJsdWVcIixcImJsYWNrXCIsIFwiZ3JlZW5cIixcInllbGxvd1wiLCBcInB1cnBsZVwiLCBcInJlZFwiKSkgK1xuICAgIHRoZW1lKGF4aXMudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSxsZWdlbmQudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSwgbGVnZW5kLnRleHQgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE0KSwgdGl0bGUgPSAgZWxlbWVudF90ZXh0KHNpemUgPSAxOCwgZmFjZSA9IFwiYm9sZFwiKSkgK1xuICAgIHhsYWIoXCJFbGVjdGlvblwiKSArXG4gICAgeWxhYihcIlRGU2NhbGUgLSBEaW0gMlwiKVxuXG5wbG90X2NtcGxvZyA8LSBnZ3Bsb3QoZGF0YSA9IGRmLFxuICAgICAgICAgICAgICAgICAgYWVzKHg9ZmFjdG9yKGVsZWN0aW9uKSwgeT1scl9sb2csIGNvbG9yPXBhcnR5LGdyb3VwPXBhcnR5KSkgK1xuICAgIGdlb21fcG9pbnQoKSArXG4gICAgZ2VvbV9saW5lKGFscGhhPTAuNSwgc2l6ZSA9IDEpICsgXG4gICAgc2NhbGVfY29sb3JfbWFudWFsKG5hbWUgPSBcIlBhcnR5XCIsICBcbiAgICAgICAgICAgICAgICAgICAgICAgdmFsdWVzID0gYyhcImJsdWVcIixcImJsYWNrXCIsIFwiZ3JlZW5cIixcInllbGxvd1wiLCBcInB1cnBsZVwiLCBcInJlZFwiKSkgK1xuICAgIHRoZW1lKGF4aXMudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSxsZWdlbmQudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSwgbGVnZW5kLnRleHQgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE0KSwgdGl0bGUgPSAgZWxlbWVudF90ZXh0KHNpemUgPSAxOCwgZmFjZSA9IFwiYm9sZFwiKSkgK1xuICAgIHhsYWIoXCJFbGVjdGlvblwiKSArXG4gICAgeWxhYihcIkNNUCAtIExvZ1wiKVxuXG5cblxuXG5saWJyYXJ5KHJhZ2cpXG5saWJyYXJ5KGdncHVicilcbmZpZ3VyZV8xIDwtIGdnYXJyYW5nZShwbG90X3RmMSwgcGxvdF93ZiwgcGxvdF9kMnYxLCBwbG90X3RmMiwgcGxvdF9jbXBsb2csIHBsb3RfZDJ2MixcbiAgICAgICAgICAgICAgICAgICAgICBjb21tb24ubGVnZW5kID0gVFJVRSlcblxuYWdnX3BuZyhmaWxlbmFtZSA9IGhlcmU6OmhlcmUoXCJyZXN1bHRzXCIsXCJ0YWJzIGFuZCBmaWdzXCIsXCJmaWd1cmUgMS5wbmdcIiksIHJlcyA9IDM2MCwgd2lkdGggPSA0ODAwLCBoZWlnaHQgPSAzMjAwKVxuZmlndXJlXzFcbmludmlzaWJsZShkZXYub2ZmKCkpXG5gYGAifQ== -->

```r
plot_wf <- ggplot(data = df,
                  aes(x=factor(election), y=wf_score, color=party,group=party)) +
    geom_point() +
    geom_line(alpha=0.5, linewidth = 1) + 
    scale_color_manual(name = "Party",  
                       values = c("blue","black", "green","yellow", "purple", "red")) +
    theme(axis.title = element_text(size = 15),legend.title = element_text(size = 15), legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold")) +
    xlab("Election") +
    ylab("Wordfish")


plot_d2v1 <- ggplot(data = df,
                  aes(x=factor(election), y= d2v_d1, color=party,group=party)) +
    geom_point() +
    geom_line(alpha=0.5, size = 1) + 
    scale_color_manual(name = "Party",  
                       values = c("blue","black", "green","yellow", "purple", "red")) +
    theme(axis.title = element_text(size = 15),legend.title = element_text(size = 15), legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold")) +
    xlab("Election") +
    ylab("Doc2Vec - Dim 1")

plot_d2v2 <- ggplot(data = df,
                  aes(x=factor(election), y= d2v_d2, color=party,group=party)) +
    geom_point() +
    geom_line(alpha=0.5, size = 1) + 
    scale_color_manual(name = "Party",  
                       values = c("blue","black", "green","yellow", "purple", "red")) +
    theme(axis.title = element_text(size = 15),legend.title = element_text(size = 15), legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold")) +
    xlab("Election") +
    ylab("Doc2Vec - Dim 2")


plot_tf1 <- ggplot(data = df,
                  aes(x=factor(election), y=umap_d1, color=party,group=party)) +
    geom_point() +
    geom_line(alpha=0.5, size = 1) + 
    scale_color_manual(name = "Party",  
                       values = c("blue","black", "green","yellow", "purple", "red")) +
    theme(axis.title = element_text(size = 15),legend.title = element_text(size = 15), legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold")) +
    xlab("Election") +
    ylab("TFScale - Dim 1")


plot_tf2 <- ggplot(data = df,
                  aes(x=factor(election), y=umap_d2, color=party,group=party)) +
    geom_point() +
    geom_line(alpha=0.5, size = 1) + 
    scale_color_manual(name = "Party",  
                       values = c("blue","black", "green","yellow", "purple", "red")) +
    theme(axis.title = element_text(size = 15),legend.title = element_text(size = 15), legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold")) +
    xlab("Election") +
    ylab("TFScale - Dim 2")

plot_cmplog <- ggplot(data = df,
                  aes(x=factor(election), y=lr_log, color=party,group=party)) +
    geom_point() +
    geom_line(alpha=0.5, size = 1) + 
    scale_color_manual(name = "Party",  
                       values = c("blue","black", "green","yellow", "purple", "red")) +
    theme(axis.title = element_text(size = 15),legend.title = element_text(size = 15), legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold")) +
    xlab("Election") +
    ylab("CMP - Log")




library(ragg)
library(ggpubr)
figure_1 <- ggarrange(plot_tf1, plot_wf, plot_d2v1, plot_tf2, plot_cmplog, plot_d2v2,
                      common.legend = TRUE)

agg_png(filename = here::here("results","tabs and figs","figure 1.png"), res = 360, width = 4800, height = 3200)
figure_1
invisible(dev.off())
```

<!-- rnb-source-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuTkFcbk5BXG5OQVxuYGBgIn0= -->

```r
NA
NA
NA
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->


## Correlation table

<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuY29yX3RhYmxlIDwtIHJvdW5kKGNvcl90YWJsZSwgMilcblxuYGBgIn0= -->

```r
cor_table <- round(cor_table, 2)

```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiRXJyb3IgaW4gTWF0aC5kYXRhLmZyYW1lKGxpc3QodmFyMSA9IGMoXCJscl9sb2dcIiwgXCJscl9sb2dcIiwgXCJscl9sb2dcIiwgXCJscl9sb2dcIiwgIDogXG4gIG5vbi1udW1lcmljLWFsaWtlIHZhcmlhYmxlKHMpIGluIGRhdGEgZnJhbWU6IHZhcjEsIHZhcjIsIG1ldGhvZFxuIn0= -->

```
Error in Math.data.frame(list(var1 = c("lr_log", "lr_log", "lr_log", "lr_log",  : 
  non-numeric-alike variable(s) in data frame: var1, var2, method
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



## Wordfish - Welfare


<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuZGZtYXQgPC0gY29ycHVzKGdyb3VwZWRfdGV4dHMsIHRleHRfZmllbGQgPSBcInRleHRfbmV3XCIpICU+JVxuICAgICAgICAgICAgIHRva2VucyhyZW1vdmVfcHVuY3QgPSBUUlVFLCByZW1vdmVfbnVtYmVycyA9IFRSVUUsIHJlbW92ZV9zeW1ib2xzID0gVFJVRSkgJT4lXG4gICAgICAgICAgICAgdG9rZW5zX3JlbW92ZShzdG9wd29yZHMoXCJkZVwiKSkgJT4lXG4gICAgICAgICAgICAgZGZtKCkgJT4lXG4gICAgICAgICAgICAgZGZtX3RyaW0obWluX3Rlcm1mcmVxID0gNTAsIHZlcmJvc2UgPSBUUlVFLCBtaW5fZG9jZnJlcSA9IDAuMDMsIGRvY2ZyZXFfdHlwZSA9IFwicHJvcFwiKSAlPiVcbiAgICAgICAgICAgICBkZm1fc3Vic2V0KG50b2tlbiguKSA+IDAsIGRyb3BfZG9jaWQgPSBUUlVFKVxuXG5gYGAifQ== -->

```r
dfmat <- corpus(grouped_texts, text_field = "text_new") %>%
             tokens(remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE) %>%
             tokens_remove(stopwords("de")) %>%
             dfm() %>%
             dfm_trim(min_termfreq = 50, verbose = TRUE, min_docfreq = 0.03, docfreq_type = "prop") %>%
             dfm_subset(ntoken(.) > 0, drop_docid = TRUE)

```

<!-- rnb-source-end -->

<!-- rnb-output-begin eyJkYXRhIjoiUmVtb3ZpbmcgZmVhdHVyZXMgb2NjdXJyaW5nOiBcbiAgLSBmZXdlciB0aGFuIDUwIHRpbWVzOiAxMCwxMjNcbiAgLSBpbiBmZXdlciB0aGFuIDEuMTQgZG9jdW1lbnRzOiA2LDA1MVxuICBUb3RhbCBmZWF0dXJlcyByZW1vdmVkOiAxMCwxMjMgKDk5LjAlKS5cbiJ9 -->

```
Removing features occurring: 
  - fewer than 50 times: 10,123
  - in fewer than 1.14 documents: 6,051
  Total features removed: 10,123 (99.0%).
```



<!-- rnb-output-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->

## Figure 3


<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxucGxvdF93ZiA8LSBnZ3Bsb3QoZGF0YSA9IGRmX3dmLFxuICAgICAgICAgICAgICAgICAgYWVzKHg9ZmFjdG9yKGVsZWN0aW9uKSwgeT13Zl9zY29yZSwgY29sb3I9cGFydHksZ3JvdXA9cGFydHkpKSArXG4gICAgZ2VvbV9wb2ludCgpICtcbiAgICBnZW9tX2xpbmUoYWxwaGE9MC41LCBsaW5ld2lkdGggPSAxKSArIFxuICAgIHNjYWxlX2NvbG9yX21hbnVhbChuYW1lID0gXCJQYXJ0eVwiLCAgXG4gICAgICAgICAgICAgICAgICAgICAgIHZhbHVlcyA9IGMoXCJibHVlXCIsXCJibGFja1wiLCBcImdyZWVuXCIsXCJ5ZWxsb3dcIiwgXCJwdXJwbGVcIiwgXCJyZWRcIikpICtcbiAgICB0aGVtZShheGlzLnRpdGxlID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNSksbGVnZW5kLnRpdGxlID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNSksIGxlZ2VuZC50ZXh0ID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNCksIHRpdGxlID0gIGVsZW1lbnRfdGV4dChzaXplID0gMTgsIGZhY2UgPSBcImJvbGRcIikpICtcbiAgICB4bGFiKFwiRWxlY3Rpb25cIikgK1xuICAgIHlsYWIoXCJXb3JkZmlzaFwiKVxuXG5cbnBsb3RfZDJ2MSA8LSBnZ3Bsb3QoZGF0YSA9IGRmX3dmLFxuICAgICAgICAgICAgICAgICAgYWVzKHg9ZmFjdG9yKGVsZWN0aW9uKSwgeT0gZDJ2X2QxLCBjb2xvcj1wYXJ0eSxncm91cD1wYXJ0eSkpICtcbiAgICBnZW9tX3BvaW50KCkgK1xuICAgIGdlb21fbGluZShhbHBoYT0wLjUsIHNpemUgPSAxKSArIFxuICAgIHNjYWxlX2NvbG9yX21hbnVhbChuYW1lID0gXCJQYXJ0eVwiLCAgXG4gICAgICAgICAgICAgICAgICAgICAgIHZhbHVlcyA9IGMoXCJibHVlXCIsXCJibGFja1wiLCBcImdyZWVuXCIsXCJ5ZWxsb3dcIiwgXCJwdXJwbGVcIiwgXCJyZWRcIikpICtcbiAgICB0aGVtZShheGlzLnRpdGxlID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNSksbGVnZW5kLnRpdGxlID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNSksIGxlZ2VuZC50ZXh0ID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNCksIHRpdGxlID0gIGVsZW1lbnRfdGV4dChzaXplID0gMTgsIGZhY2UgPSBcImJvbGRcIikpICtcbiAgICB4bGFiKFwiRWxlY3Rpb25cIikgK1xuICAgIHlsYWIoXCJEb2MyVmVjIC0gRGltIDFcIilcblxuXG5cbnBsb3RfdGYxIDwtIGdncGxvdChkYXRhID0gZGZfd2YsXG4gICAgICAgICAgICAgICAgICBhZXMoeD1mYWN0b3IoZWxlY3Rpb24pLCB5PXVtYXBfZDEsIGNvbG9yPXBhcnR5LGdyb3VwPXBhcnR5KSkgK1xuICAgIGdlb21fcG9pbnQoKSArXG4gICAgZ2VvbV9saW5lKGFscGhhPTAuNSwgc2l6ZSA9IDEpICsgXG4gICAgc2NhbGVfY29sb3JfbWFudWFsKG5hbWUgPSBcIlBhcnR5XCIsICBcbiAgICAgICAgICAgICAgICAgICAgICAgdmFsdWVzID0gYyhcImJsdWVcIixcImJsYWNrXCIsIFwiZ3JlZW5cIixcInllbGxvd1wiLCBcInB1cnBsZVwiLCBcInJlZFwiKSkgK1xuICAgIHRoZW1lKGF4aXMudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSxsZWdlbmQudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSwgbGVnZW5kLnRleHQgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE0KSwgdGl0bGUgPSAgZWxlbWVudF90ZXh0KHNpemUgPSAxOCwgZmFjZSA9IFwiYm9sZFwiKSkgK1xuICAgIHhsYWIoXCJFbGVjdGlvblwiKSArXG4gICAgeWxhYihcIlRGU2NhbGUgLSBEaW0gMVwiKVxuXG5cblxuXG5wbG90X2NtcGxvZyA8LSBnZ3Bsb3QoZGF0YSA9IGRmX3dmLFxuICAgICAgICAgICAgICAgICAgYWVzKHg9ZmFjdG9yKGVsZWN0aW9uKSwgeT1scl9sb2csIGNvbG9yPXBhcnR5LGdyb3VwPXBhcnR5KSkgK1xuICAgIGdlb21fcG9pbnQoKSArXG4gICAgZ2VvbV9saW5lKGFscGhhPTAuNSwgc2l6ZSA9IDEpICsgXG4gICAgc2NhbGVfY29sb3JfbWFudWFsKG5hbWUgPSBcIlBhcnR5XCIsICBcbiAgICAgICAgICAgICAgICAgICAgICAgdmFsdWVzID0gYyhcImJsdWVcIixcImJsYWNrXCIsIFwiZ3JlZW5cIixcInllbGxvd1wiLCBcInB1cnBsZVwiLCBcInJlZFwiKSkgK1xuICAgIHRoZW1lKGF4aXMudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSxsZWdlbmQudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSwgbGVnZW5kLnRleHQgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE0KSwgdGl0bGUgPSAgZWxlbWVudF90ZXh0KHNpemUgPSAxOCwgZmFjZSA9IFwiYm9sZFwiKSkgK1xuICAgIHhsYWIoXCJFbGVjdGlvblwiKSArXG4gICAgeWxhYihcIkNNUCAtIExvZ1wiKVxuXG5wbG90X2NtcGFicyA8LSBnZ3Bsb3QoZGF0YSA9IGRmX3dmLFxuICAgICAgICAgICAgICAgICAgYWVzKHg9ZmFjdG9yKGVsZWN0aW9uKSwgeT1scl9hYnMsIGNvbG9yPXBhcnR5LGdyb3VwPXBhcnR5KSkgK1xuICAgIGdlb21fcG9pbnQoKSArXG4gICAgZ2VvbV9saW5lKGFscGhhPTAuNSwgc2l6ZSA9IDEpICsgXG4gICAgc2NhbGVfY29sb3JfbWFudWFsKG5hbWUgPSBcIlBhcnR5XCIsICBcbiAgICAgICAgICAgICAgICAgICAgICAgdmFsdWVzID0gYyhcImJsdWVcIixcImJsYWNrXCIsIFwiZ3JlZW5cIixcInllbGxvd1wiLCBcInB1cnBsZVwiLCBcInJlZFwiKSkgK1xuICAgIHRoZW1lKGF4aXMudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSxsZWdlbmQudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSwgbGVnZW5kLnRleHQgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE0KSwgdGl0bGUgPSAgZWxlbWVudF90ZXh0KHNpemUgPSAxOCwgZmFjZSA9IFwiYm9sZFwiKSkgK1xuICAgIHhsYWIoXCJFbGVjdGlvblwiKSArXG4gICAgeWxhYihcIkNNUCAtIE9yZ1wiKVxuXG5cbmxlZ2VuZF8yIDwtIGdldF9sZWdlbmQocGxvdF90ZjEpXG5maWd1cmVfMiA8LSBnZ2FycmFuZ2UocGxvdF90ZjEsIHBsb3Rfd2YsIHBsb3RfZDJ2MSwgcGxvdF9jbXBsb2csIHBsb3RfY21wYWJzLCBhc19nZ3Bsb3QobGVnZW5kXzIpLFxuICAgICAgICAgICAgICAgICAgICAgIGxlZ2VuZD1cIm5vbmVcIilcbmFnZ19wbmcoZmlsZW5hbWUgPSBoZXJlOjpoZXJlKFwicmVzdWx0c1wiLFwidGFicyBhbmQgZmlnc1wiLFwiZmlndXJlIDIucG5nXCIpLCByZXMgPSAzNjAsIHdpZHRoID0gNDgwMCwgaGVpZ2h0ID0gMzIwMClcbmZpZ3VyZV8yXG5pbnZpc2libGUoZGV2Lm9mZigpKVxuYGBgIn0= -->

```r
plot_wf <- ggplot(data = df_wf,
                  aes(x=factor(election), y=wf_score, color=party,group=party)) +
    geom_point() +
    geom_line(alpha=0.5, linewidth = 1) + 
    scale_color_manual(name = "Party",  
                       values = c("blue","black", "green","yellow", "purple", "red")) +
    theme(axis.title = element_text(size = 15),legend.title = element_text(size = 15), legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold")) +
    xlab("Election") +
    ylab("Wordfish")


plot_d2v1 <- ggplot(data = df_wf,
                  aes(x=factor(election), y= d2v_d1, color=party,group=party)) +
    geom_point() +
    geom_line(alpha=0.5, size = 1) + 
    scale_color_manual(name = "Party",  
                       values = c("blue","black", "green","yellow", "purple", "red")) +
    theme(axis.title = element_text(size = 15),legend.title = element_text(size = 15), legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold")) +
    xlab("Election") +
    ylab("Doc2Vec - Dim 1")



plot_tf1 <- ggplot(data = df_wf,
                  aes(x=factor(election), y=umap_d1, color=party,group=party)) +
    geom_point() +
    geom_line(alpha=0.5, size = 1) + 
    scale_color_manual(name = "Party",  
                       values = c("blue","black", "green","yellow", "purple", "red")) +
    theme(axis.title = element_text(size = 15),legend.title = element_text(size = 15), legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold")) +
    xlab("Election") +
    ylab("TFScale - Dim 1")




plot_cmplog <- ggplot(data = df_wf,
                  aes(x=factor(election), y=lr_log, color=party,group=party)) +
    geom_point() +
    geom_line(alpha=0.5, size = 1) + 
    scale_color_manual(name = "Party",  
                       values = c("blue","black", "green","yellow", "purple", "red")) +
    theme(axis.title = element_text(size = 15),legend.title = element_text(size = 15), legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold")) +
    xlab("Election") +
    ylab("CMP - Log")

plot_cmpabs <- ggplot(data = df_wf,
                  aes(x=factor(election), y=lr_abs, color=party,group=party)) +
    geom_point() +
    geom_line(alpha=0.5, size = 1) + 
    scale_color_manual(name = "Party",  
                       values = c("blue","black", "green","yellow", "purple", "red")) +
    theme(axis.title = element_text(size = 15),legend.title = element_text(size = 15), legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold")) +
    xlab("Election") +
    ylab("CMP - Org")


legend_2 <- get_legend(plot_tf1)
figure_2 <- ggarrange(plot_tf1, plot_wf, plot_d2v1, plot_cmplog, plot_cmpabs, as_ggplot(legend_2),
                      legend="none")
agg_png(filename = here::here("results","tabs and figs","figure 2.png"), res = 360, width = 4800, height = 3200)
figure_2
invisible(dev.off())
```

<!-- rnb-source-end -->

<!-- rnb-plot-begin eyJoZWlnaHQiOjQzMi42MzI5LCJ3aWR0aCI6NzAwLCJzaXplX2JlaGF2aW9yIjowLCJjb25kaXRpb25zIjpbXX0= -->

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABNoAAAL/CAMAAACH5FrHAAAAA1BMVEX///+nxBvIAAAACXBIWXMAACE3AAAhNwEzWJ96AAADs0lEQVR4nO3BMQEAAADCoPVPbQlPoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgIsBjPcAAU6Ot3wAAAAASUVORK5CYII=" />

<!-- rnb-plot-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuTkFcbk5BXG5gYGAifQ== -->

```r
NA
NA
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->


## Figure 4


<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxubHJfbG9nIDwtIGRmX2NtcFssXCJscl9sb2dcIl1cbmRmX3NjYWxlZCA8LSBiaW5kX2NvbHMocGNhX3NjYWxlZCwgdW1hcF9zX3NjYWxlZCwgdW1hcF91c19zY2FsZWQsIGl2aXNfc2NhbGVkLCBsZGFfc2NhbGVkLCBhZV9zY2FsZWQpXG5jb3JfdGFibGVfMiA8LSBkZl9zY2FsZWQgJT4lIGNvcl90ZXN0KHZhcnMgPSBcInBjYV9kMVwiLCB2YXJzMj1jKFwicGNhX2QxXCIsXCJ1bWFwX3NfZDFcIiwgXCJ1bWFwX3VzX2QxXCIsIFwiaXZpc19kMVwiLCBcImxkYV9kMVwiLCBcImFlX2QxXCIpKVxuY29yXzIgPC0gYWJzKGNvcl90YWJsZV8yW1wiY29yXCJdKVxuZGZfZW1iZWRzIDwtIGJpbmRfY29scyhwY2FfZW1iZWRzLCB1bWFwX3NfZW1iZWRzLCB1bWFwX3VzX2VtYmVkcywgaXZpc19lbWJlZHMsIGxkYV9lbWJlZHMsIGFlX2VtYmVkcywgbHJfbG9nKVxuY29yX3RhYmxlXzMgPC0gZGZfZW1iZWRzICU+JSBjb3JfdGVzdCh2YXJzID0gXCJscl9sb2dcIiwgdmFyczI9YyhcInBjYV9kMVwiLFwidW1hcF9zX2QxXCIsIFwidW1hcF91c19kMVwiLCBcIml2aXNfZDFcIiwgXCJsZGFfZDFcIiwgXCJhZV9kMVwiKSlcbmNvcl8zIDwtIGFicyhjb3JfdGFibGVfM1tcImNvclwiXSlcblxuZGZfZHIgPC0gYmluZF9jb2xzKGRyX3ZhbGlkLCBjb3JfMiwgY29yXzMpXG5jb2xuYW1lcyhkZl9kcikgPC0gYyhcInRlY2huaXF1ZXNcIixcInR3XCIsXCJzaWxcIixcImNvcl9wY2FcIiwgXCJjb3JfY21wXCIpXG5cbnBsb3RfdHcgPC0gZ2dwbG90KGRhdGEgPSBkZl9kcixcbiAgICAgICAgICAgICAgICAgIGFlcyh4PXRlY2huaXF1ZXMsIHk9dHcsIGZpbGwgPSB0ZWNobmlxdWVzKSkgK1xuICAgIGdlb21fYmFyKHN0YXQ9XCJpZGVudGl0eVwiKSArXG4gICAgc2NhbGVfZmlsbF92aXJpZGlzKG5hbWUgPSBcIlRlY2huaXF1ZXNcIiwgIFxuICAgICAgICAgICAgICAgICAgICAgICBsYWJlbHMgPSBjKFwiQXV0b2VuY29kZXJzXCIsIFwiSXZpc1wiLCBcIkxEQVwiLCBcIlBDQVwiLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFwiVU1BUCAtIFN1cGVydmlzZWRcIixcIlVNQVAgLSBVbnN1cGVydmlzZWRcIiksXG4gICAgICAgICAgICAgICAgICAgICAgZGlzY3JldGUgPSBUUlVFKSArXG4gICAgdGhlbWUoYXhpcy50aXRsZS55ID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNSksbGVnZW5kLnRpdGxlID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNSksIFxuICAgICAgICAgIGxlZ2VuZC50ZXh0ID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNCksIHRpdGxlID0gIGVsZW1lbnRfdGV4dChzaXplID0gMTgsIGZhY2UgPSBcImJvbGRcIiksXG4gICAgICAgICAgYXhpcy50ZXh0Lng9ZWxlbWVudF9ibGFuaygpLCBheGlzLnRpY2tzLng9ZWxlbWVudF9ibGFuaygpXG4gICAgICAgICAgKSArXG4gICAgeGxhYihcIlRydXN0d29ydGhpbmVzc1wiKSArXG4gICAgeWxhYihcIkNvZWZmaWNpZW50c1wiKVxuXG5wbG90X3NpbCA8LSBnZ3Bsb3QoZGF0YSA9IGRmX2RyLFxuICAgICAgICAgICAgICAgICAgYWVzKHg9dGVjaG5pcXVlcywgeT1zaWwsIGZpbGwgPSB0ZWNobmlxdWVzKSkgK1xuICAgIGdlb21fYmFyKHN0YXQ9XCJpZGVudGl0eVwiKSArXG4gICAgc2NhbGVfZmlsbF92aXJpZGlzKG5hbWUgPSBcIlRlY2huaXF1ZXNcIiwgICBcbiAgICAgICAgICAgICAgICAgICAgICAgbGFiZWxzID0gYyhcIkF1dG9lbmNvZGVyc1wiLCBcIkl2aXNcIiwgXCJMREFcIiwgXCJQQ0FcIixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBcIlVNQVAgLSBTdXBlcnZpc2VkXCIsXCJVTUFQIC0gVW5zdXBlcnZpc2VkXCIpLCBkaXNjcmV0ZT1UUlVFKSArXG4gICAgdGhlbWUoYXhpcy50aXRsZS55ID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNSksbGVnZW5kLnRpdGxlID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNSksIFxuICAgICAgICAgIGxlZ2VuZC50ZXh0ID0gZWxlbWVudF90ZXh0KHNpemUgPSAxNCksIHRpdGxlID0gIGVsZW1lbnRfdGV4dChzaXplID0gMTgsIGZhY2UgPSBcImJvbGRcIiksXG4gICAgICAgICAgYXhpcy50ZXh0Lng9ZWxlbWVudF9ibGFuaygpLCBheGlzLnRpY2tzLng9ZWxlbWVudF9ibGFuaygpXG4gICAgICAgICAgKSArXG4gICAgeGxhYihcIlNpbGhvdWV0dGUgc2NvcmVzXCIpICtcbiAgICB5bGFiKFwiQ29lZmZpY2llbnRzXCIpXG5cbnBsb3RfY29ycGNhIDwtIGdncGxvdChkYXRhID0gZGZfZHIsXG4gICAgICAgICAgICAgICAgICBhZXMoeD10ZWNobmlxdWVzLCB5PWNvcl9wY2EsIGZpbGwgPSB0ZWNobmlxdWVzKSkgK1xuICAgIGdlb21fYmFyKHN0YXQ9XCJpZGVudGl0eVwiKSArXG4gICAgc2NhbGVfZmlsbF92aXJpZGlzKG5hbWUgPSBcIlRlY2huaXF1ZXNcIiwgIFxuICAgICAgICAgICAgICAgICAgICAgICBsYWJlbHMgPSBjKFwiQXV0b2VuY29kZXJzXCIsIFwiSXZpc1wiLCBcIkxEQVwiLCBcIlBDQVwiLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFwiVU1BUCAtIFN1cGVydmlzZWRcIixcIlVNQVAgLSBVbnN1cGVydmlzZWRcIiksIGRpc2NyZXRlPVRSVUUpICtcbiAgICB0aGVtZShheGlzLnRpdGxlLnkgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSxsZWdlbmQudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSwgXG4gICAgICAgICAgbGVnZW5kLnRleHQgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE0KSwgdGl0bGUgPSAgZWxlbWVudF90ZXh0KHNpemUgPSAxOCwgZmFjZSA9IFwiYm9sZFwiKSxcbiAgICAgICAgICBheGlzLnRleHQueD1lbGVtZW50X2JsYW5rKCksIGF4aXMudGlja3MueD1lbGVtZW50X2JsYW5rKClcbiAgICAgICAgICApICtcbiAgICB4bGFiKFwiQWJzb2x1dGUgQ29ycmVsYXRpb24gdG8gUENBIGVtYmVkZGluZ3NcIikgK1xuICAgIHlsYWIoXCJDb2VmZmljaWVudHNcIilcblxucGxvdF9jb3JjbXAgPC0gZ2dwbG90KGRhdGEgPSBkZl9kcixcbiAgICAgICAgICAgICAgICAgIGFlcyh4PXRlY2huaXF1ZXMsIHk9Y29yX2NtcCwgZmlsbCA9IHRlY2huaXF1ZXMpKSArXG4gICAgZ2VvbV9iYXIoc3RhdD1cImlkZW50aXR5XCIpICtcbiAgICBzY2FsZV9maWxsX3ZpcmlkaXMobmFtZSA9IFwiVGVjaG5pcXVlc1wiLCAgXG4gICAgICAgICAgICAgICAgICAgICAgIGxhYmVscyA9IGMoXCJBdXRvZW5jb2RlcnNcIiwgXCJJdmlzXCIsIFwiTERBXCIsIFwiUENBXCIsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgXCJVTUFQIC0gU3VwZXJ2aXNlZFwiLFwiVU1BUCAtIFVuc3VwZXJ2aXNlZFwiKSwgZGlzY3JldGU9VFJVRSkgK1xuICAgIHRoZW1lKGF4aXMudGl0bGUueSA9IGVsZW1lbnRfdGV4dChzaXplID0gMTUpLGxlZ2VuZC50aXRsZSA9IGVsZW1lbnRfdGV4dChzaXplID0gMTUpLCBcbiAgICAgICAgICBsZWdlbmQudGV4dCA9IGVsZW1lbnRfdGV4dChzaXplID0gMTQpLCB0aXRsZSA9ICBlbGVtZW50X3RleHQoc2l6ZSA9IDE4LCBmYWNlID0gXCJib2xkXCIpLFxuICAgICAgICAgIGF4aXMudGV4dC54PWVsZW1lbnRfYmxhbmsoKSwgYXhpcy50aWNrcy54PWVsZW1lbnRfYmxhbmsoKVxuICAgICAgICAgICkgK1xuICAgIHhsYWIoXCJBYnNvbHV0ZSBDb3JyZWxhdGlvbiB0byBDTVAgbG9nIHNjb3Jlc1wiKSArXG4gICAgeWxhYihcIkNvZWZmaWNpZW50c1wiKVxuXG5cbmZpZ3VyZV8zIDwtIGdnYXJyYW5nZShwbG90X3R3LCBwbG90X3NpbCwgcGxvdF9jb3JwY2EsIHBsb3RfY29yY21wLCBcbiAgICAgICAgICAgICAgICAgICAgICBjb21tb24ubGVnZW5kPVRSVUUpXG5hZ2dfcG5nKGZpbGVuYW1lID0gaGVyZTo6aGVyZShcInJlc3VsdHNcIixcInRhYnMgYW5kIGZpZ3NcIixcImZpZ3VyZSAzLnBuZ1wiKSwgcmVzID0gMzYwLCB3aWR0aCA9IDQ4MDAsIGhlaWdodCA9IDMyMDApXG5maWd1cmVfM1xuaW52aXNpYmxlKGRldi5vZmYoKSlcblxuYGBgIn0= -->

```r
lr_log <- df_cmp[,"lr_log"]
df_scaled <- bind_cols(pca_scaled, umap_s_scaled, umap_us_scaled, ivis_scaled, lda_scaled, ae_scaled)
cor_table_2 <- df_scaled %>% cor_test(vars = "pca_d1", vars2=c("pca_d1","umap_s_d1", "umap_us_d1", "ivis_d1", "lda_d1", "ae_d1"))
cor_2 <- abs(cor_table_2["cor"])
df_embeds <- bind_cols(pca_embeds, umap_s_embeds, umap_us_embeds, ivis_embeds, lda_embeds, ae_embeds, lr_log)
cor_table_3 <- df_embeds %>% cor_test(vars = "lr_log", vars2=c("pca_d1","umap_s_d1", "umap_us_d1", "ivis_d1", "lda_d1", "ae_d1"))
cor_3 <- abs(cor_table_3["cor"])

df_dr <- bind_cols(dr_valid, cor_2, cor_3)
colnames(df_dr) <- c("techniques","tw","sil","cor_pca", "cor_cmp")

plot_tw <- ggplot(data = df_dr,
                  aes(x=techniques, y=tw, fill = techniques)) +
    geom_bar(stat="identity") +
    scale_fill_viridis(name = "Techniques",  
                       labels = c("Autoencoders", "Ivis", "LDA", "PCA",
                                  "UMAP - Supervised","UMAP - Unsupervised"),
                      discrete = TRUE) +
    theme(axis.title.y = element_text(size = 15),legend.title = element_text(size = 15), 
          legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold"),
          axis.text.x=element_blank(), axis.ticks.x=element_blank()
          ) +
    xlab("Trustworthiness") +
    ylab("Coefficients")

plot_sil <- ggplot(data = df_dr,
                  aes(x=techniques, y=sil, fill = techniques)) +
    geom_bar(stat="identity") +
    scale_fill_viridis(name = "Techniques",   
                       labels = c("Autoencoders", "Ivis", "LDA", "PCA",
                                  "UMAP - Supervised","UMAP - Unsupervised"), discrete=TRUE) +
    theme(axis.title.y = element_text(size = 15),legend.title = element_text(size = 15), 
          legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold"),
          axis.text.x=element_blank(), axis.ticks.x=element_blank()
          ) +
    xlab("Silhouette scores") +
    ylab("Coefficients")

plot_corpca <- ggplot(data = df_dr,
                  aes(x=techniques, y=cor_pca, fill = techniques)) +
    geom_bar(stat="identity") +
    scale_fill_viridis(name = "Techniques",  
                       labels = c("Autoencoders", "Ivis", "LDA", "PCA",
                                  "UMAP - Supervised","UMAP - Unsupervised"), discrete=TRUE) +
    theme(axis.title.y = element_text(size = 15),legend.title = element_text(size = 15), 
          legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold"),
          axis.text.x=element_blank(), axis.ticks.x=element_blank()
          ) +
    xlab("Absolute Correlation to PCA embeddings") +
    ylab("Coefficients")

plot_corcmp <- ggplot(data = df_dr,
                  aes(x=techniques, y=cor_cmp, fill = techniques)) +
    geom_bar(stat="identity") +
    scale_fill_viridis(name = "Techniques",  
                       labels = c("Autoencoders", "Ivis", "LDA", "PCA",
                                  "UMAP - Supervised","UMAP - Unsupervised"), discrete=TRUE) +
    theme(axis.title.y = element_text(size = 15),legend.title = element_text(size = 15), 
          legend.text = element_text(size = 14), title =  element_text(size = 18, face = "bold"),
          axis.text.x=element_blank(), axis.ticks.x=element_blank()
          ) +
    xlab("Absolute Correlation to CMP log scores") +
    ylab("Coefficients")


figure_3 <- ggarrange(plot_tw, plot_sil, plot_corpca, plot_corcmp, 
                      common.legend=TRUE)
agg_png(filename = here::here("results","tabs and figs","figure 3.png"), res = 360, width = 4800, height = 3200)
figure_3
invisible(dev.off())

```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->



## Correlation - hypothetical tests

<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->



<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->




## Figure 5

<!-- rnb-text-end -->


<!-- rnb-chunk-begin -->


<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuaWRlbyA8LSBjKFwiQ29uc2VydmF0aXZlXCIsIFwiTGliZXJhbFwiLCBcIk1vZGVyYXRlXCIsIFwiTm90IHN1cmVcIixcbiAgICAgICAgICBcIlZlcnkgY29uc2VydmF0aXZlXCIsIFwiVmVyeSBsaWJlcmFsXCIpXG5cbnRydW1wX2VtYmVkc19hZ2cgPC0gdHJ1bXBfZW1iZWRzICU+JVxuICAgIGJpbmRfY29scyh0cnVtcF9lbWJlZHNfMzApICU+JVxuICAgIG11dGF0ZShpZGVvNT1pZGVvKSAlPiVcbiAgICBmaWx0ZXIoaWRlbzUgIT0gXCJOb3Qgc3VyZVwiKVxuXG5kZl90cnVtcCA8LSB0cnVtcF9zY2FsZWQgJT4lXG4gICAgYmluZF9jb2xzKHRydW1wX3NjYWxlZF8zMCkgJT4lXG4gICAgbXV0YXRlKGlkZW81PXRydW1wX3R3ZWV0cyRpZGVvNSkgJT4lXG4gICAgZmlsdGVyKGlkZW81ICE9IFwiTm90IHN1cmVcIilcblxucGxvdF9pZGVvIDwtIGdncGxvdChkYXRhID0gZGZfdHJ1bXAsXG4gICAgICAgICAgICAgICAgICBhZXMoeD1lbWJfZDEsIHk9ZW1iX2QyLCBcbiAgICAgICAgICAgICAgICAgICAgICBjb2xvciA9IGZhY3RvcihpZGVvNSkpKSArXG4gICAgZ2VvbV9wb2ludChhbHBoYT0wLjMpICtcbiAgICBnZW9tX3BvaW50KGRhdGE9dHJ1bXBfZW1iZWRzX2FnZywgc2l6ZSA9IDYpICtcbiAgICBzY2FsZV9jb2xvcl92aXJpZGlzKG5hbWUgPSBcIk9iamVjdGl2ZSBQb2xpdGljYWwgSWRlb2xvZ3lcIiwgXG4gICAgICAgICAgICAgICAgICAgICAgICBkaXNjcmV0ZSA9IFRSVUUpICtcbiAgICB0aGVtZShheGlzLnRpdGxlLnkgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSxcbiAgICAgICAgICBsZWdlbmQudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSwgXG4gICAgICAgICAgbGVnZW5kLnRleHQgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE0KSwgXG4gICAgICAgICAgdGl0bGUgPSAgZWxlbWVudF90ZXh0KHNpemUgPSAxOCwgZmFjZSA9IFwiYm9sZFwiKSxcbiAgICAgICAgICBheGlzLnRpdGxlLnggPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KVxuICAgICAgICAgICkgK1xuICAgIHhsYWIoXCJGaXJzdCBkaW1lbnNpb25cIikgK1xuICAgIHlsYWIoXCJTZWNvbmQgZGltZW5zaW9uXCIpICtcbiAgICBsYWJzKHRpdGxlID0gXCIxMDAlIGtub3duIHN0YW5jZXNcIilcbnBsb3RfaWRlb18zMCA8LSBnZ3Bsb3QoZGF0YSA9IGRmX3RydW1wLFxuICAgICAgICAgICAgICAgICAgYWVzKHg9ZW1iXzMwX2QxLCB5PWVtYl8zMF9kMiwgXG4gICAgICAgICAgICAgICAgICAgICAgY29sb3IgPSBmYWN0b3IoaWRlbzUpKSkgK1xuICAgIGdlb21fcG9pbnQoYWxwaGE9MC4zKSArXG4gICAgZ2VvbV9wb2ludChkYXRhPXRydW1wX2VtYmVkc19hZ2csIHNpemUgPSA2KSArXG4gICAgc2NhbGVfY29sb3JfdmlyaWRpcyhuYW1lID0gXCJPYmplY3RpdmUgUG9saXRpY2FsIElkZW9sb2d5XCIsXG4gICAgICAgICAgICAgICAgICAgICAgICBkaXNjcmV0ZSA9IFRSVUUpICtcbiAgICB0aGVtZShheGlzLnRpdGxlLnkgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSxcbiAgICAgICAgICBsZWdlbmQudGl0bGUgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KSwgXG4gICAgICAgICAgbGVnZW5kLnRleHQgPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE0KSwgXG4gICAgICAgICAgdGl0bGUgPSAgZWxlbWVudF90ZXh0KHNpemUgPSAxOCwgZmFjZSA9IFwiYm9sZFwiKSxcbiAgICAgICAgICBheGlzLnRpdGxlLnggPSBlbGVtZW50X3RleHQoc2l6ZSA9IDE1KVxuICAgICAgICAgICkgK1xuICAgIHhsYWIoXCJGaXJzdCBkaW1lbnNpb25cIikgK1xuICAgIHlsYWIoXCJTZWNvbmQgZGltZW5zaW9uXCIpICsgXG4gICAgbGFicyh0aXRsZSA9IFwiMzAlIGtub3duIHN0YW5jZXMgKyBDTVAgdHJhbnNmZXJyZWRcIilcblxuZmlndXJlXzUgPC0gZ2dhcnJhbmdlKHBsb3RfaWRlbywgcGxvdF9pZGVvXzMwLCBcbiAgICAgICAgICAgICAgICAgICAgICBjb21tb24ubGVnZW5kPVRSVUUpXG5hZ2dfcG5nKGZpbGVuYW1lID0gaGVyZTo6aGVyZShcInJlc3VsdHNcIixcInRhYnMgYW5kIGZpZ3NcIixcImZpZ3VyZSA1LnBuZ1wiKSwgcmVzID0gMzYwLCB3aWR0aCA9IDQ4MDAsIGhlaWdodCA9IDMyMDApXG5maWd1cmVfNVxuaW52aXNpYmxlKGRldi5vZmYoKSlcbmBgYCJ9 -->

```r
ideo <- c("Conservative", "Liberal", "Moderate", "Not sure",
          "Very conservative", "Very liberal")

trump_embeds_agg <- trump_embeds %>%
    bind_cols(trump_embeds_30) %>%
    mutate(ideo5=ideo) %>%
    filter(ideo5 != "Not sure")

df_trump <- trump_scaled %>%
    bind_cols(trump_scaled_30) %>%
    mutate(ideo5=trump_tweets$ideo5) %>%
    filter(ideo5 != "Not sure")

plot_ideo <- ggplot(data = df_trump,
                  aes(x=emb_d1, y=emb_d2, 
                      color = factor(ideo5))) +
    geom_point(alpha=0.3) +
    geom_point(data=trump_embeds_agg, size = 6) +
    scale_color_viridis(name = "Objective Political Ideology", 
                        discrete = TRUE) +
    theme(axis.title.y = element_text(size = 15),
          legend.title = element_text(size = 15), 
          legend.text = element_text(size = 14), 
          title =  element_text(size = 18, face = "bold"),
          axis.title.x = element_text(size = 15)
          ) +
    xlab("First dimension") +
    ylab("Second dimension") +
    labs(title = "100% known stances")
plot_ideo_30 <- ggplot(data = df_trump,
                  aes(x=emb_30_d1, y=emb_30_d2, 
                      color = factor(ideo5))) +
    geom_point(alpha=0.3) +
    geom_point(data=trump_embeds_agg, size = 6) +
    scale_color_viridis(name = "Objective Political Ideology",
                        discrete = TRUE) +
    theme(axis.title.y = element_text(size = 15),
          legend.title = element_text(size = 15), 
          legend.text = element_text(size = 14), 
          title =  element_text(size = 18, face = "bold"),
          axis.title.x = element_text(size = 15)
          ) +
    xlab("First dimension") +
    ylab("Second dimension") + 
    labs(title = "30% known stances + CMP transferred")

figure_5 <- ggarrange(plot_ideo, plot_ideo_30, 
                      common.legend=TRUE)
agg_png(filename = here::here("results","tabs and figs","figure 5.png"), res = 360, width = 4800, height = 3200)
figure_5
invisible(dev.off())
```

<!-- rnb-source-end -->

<!-- rnb-source-begin eyJkYXRhIjoiYGBgclxuTkFcbk5BXG5gYGAifQ== -->

```r
NA
NA
```

<!-- rnb-source-end -->

<!-- rnb-chunk-end -->


<!-- rnb-text-begin -->




<!-- rnb-text-end -->

