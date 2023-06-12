library(pdftools)
library(tidyverse)
library(quanteda)
load(here::here('data','ESW','every_single_word.RData'))
emp_bills <- pdf_text(here::here('data','dip_export','employment_7020.pdf'))
sproc_bills <- pdf_text(here::here('data','dip_export','socproc_7020.pdf'))

docs_emp <- unlist(lapply(emp_bills, stringi::stri_extract_all, 
                             regex="BT-Drucksache\\s\\d+(\\/\\d+)|BR-Drucksache\\s\\d+(\\/\\d+)"))

docs_sproc <- unlist(lapply(sproc_bills, stringi::stri_extract_all, 
                           regex="BT-Drucksache\\s\\d+(\\/\\d+)|BR-Drucksache\\s\\d+(\\/\\d+)"))

docs_emp <- stringi::stri_remove_empty_na(docs_emp)
docs_sproc <- stringi::stri_remove_empty_na(docs_sproc)

#docs_bt <- docs_extract[stringi::stri_detect(docs_extract, regex="BT\\-")]
#docs_br <- docs_extract[!stringi::stri_detect(docs_extract, regex="BT\\-")]

id_emp <- unlist(stringi::stri_extract_all(docs_emp, regex = "\\d+\\/\\d+"))
id_sproc <- unlist(stringi::stri_extract_all(docs_sproc, regex = "\\d+\\/\\d+"))

esw_emp <- every_single_word %>%
    filter(number %in% id_emp & document_type == "Gesetzentwurf") %>%
    select(number, document_type, elec_period, date, title, text)

esw_sproc <- every_single_word %>%
    filter(number %in% id_sproc & document_type == "Gesetzentwurf") %>%
    select(number, document_type, elec_period, date, title, text)



corpus_emp <- corpus(esw_emp, text_field = "text") %>% corpus_segment(pattern = "Entwurf eines Gesetz", 
                                         valuetype = "fixed", pattern_position = "after")

corpus_emp <- convert(corpus_emp, to = "data.frame") %>%
    separate_wider_delim(doc_id, ".", names = c("before", "after")) %>%
    filter(after == 1) %>%
    corpus(text_field = "text")

corp_sent <- corpus_reshape(corpus_emp, to = "sentence")

df_emp <- convert(corp_sent, to = "data.frame")
texts <- df_emp$text
seq_lens <- str_count(texts, "\\w+")
mean(seq_lens)
min(seq_lens)
max(seq_lens)
idx_rmv <- which(seq_lens<2)

df_emp <- df_emp %>% mutate(id = row_number()) %>% filter(!(id %in% idx_rmv))

write.csv(df_emp, file=here::here("data","r_outputs","esw_empl.csv"), 
          fileEncoding="UTF-8", row.names=FALSE)   


corpus_sproc <- corpus(esw_sproc, text_field = "text") %>% corpus_segment(pattern = "Entwurf eines Gesetz", 
                                                                      valuetype = "fixed", pattern_position = "after")

corpus_sproc <- convert(corpus_sproc, to = "data.frame") %>%
    separate_wider_delim(doc_id, ".", names = c("before", "after")) %>%
    filter(after == 1) %>%
    corpus(text_field = "text")

corp_sent <- corpus_reshape(corpus_sproc, to = "sentence")

df_sproc <- convert(corp_sent, to = "data.frame")
texts <- df_sproc$text
seq_lens <- str_count(texts, "\\w+")
mean(seq_lens)
min(seq_lens)
max(seq_lens)
idx_rmv <- which(seq_lens<2)

df_sproc <- df_sproc %>% mutate(id = row_number()) %>% filter(!(id %in% idx_rmv))

write.csv(df_sproc, file=here::here("data","r_outputs","esw_sproc.csv"), 
          fileEncoding="UTF-8", row.names=FALSE)   



grouped_texts <- esw_emp %>% 
    mutate(
        year = format(as.Date(date, format="%Y-%m-%d"),"%Y")
    ) %>%
    group_by(year) %>%
    summarise(
        text_new = paste(text, collapse = " "),
        num_speeches = n()
    )


dfmat <- corpus(grouped_texts, text_field = "text_new") %>%
    tokens(remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE) %>%
    tokens_remove(stopwords("de")) %>%
    dfm() %>%
    dfm_trim(min_termfreq = 50, verbose = TRUE, min_docfreq = 0.03, docfreq_type = "prop") %>%
    dfm_subset(ntoken(.) > 0, drop_docid = TRUE)

wf_mod <- quanteda.textmodels::textmodel_wordfish(dfmat, dispersion = "poisson", 
                                                  sparse = TRUE)
results <- grouped_texts %>% select(-text_new)

results$wf_score <- wf_mod$theta


write.csv(results, file=here::here("data","r_outputs","res_temp.csv"), 
          fileEncoding="UTF-8", row.names=FALSE)   
