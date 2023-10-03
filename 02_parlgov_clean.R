# install.packages("tidyverse") ## installed if not existed

library(tidyverse) 

###################################################
###################################################
###################################################
    
## Load file
election <- readr::read_csv(here::here("data/ParlGov/view_election.csv"), show_col_types = FALSE)

## Clean data
election_ger <- election %>%
    filter(election_date >= "1990-01-01" & country_name == "Germany" 
           & election_type == "parliament" 
           & !(party_name %in% c("no party affiliation", "one seat","one-seat", "etc", "none"))
           & seats != 0 & party_name_short != "AfD")  %>%
    mutate(
        year = format(as.Date(election_date), "%Y"),
        party = dplyr::recode(party_name_short, 'CDU' = 'CDU/CSU', 'CSU' = 'CDU/CSU', 'B90/Gru' = 'GRUENEN',
                              'PDS|Li' = 'DIE LINKE', 'B90/Gr' = 'GRUENEN', 'Pi' = 'Piraten', "CDU+CSU" = "CDU/CSU"),
    )  %>%
    mutate(
        bundestag = case_when(
            year == 1990 ~ 12,
            year == 1994 ~ 13,
            year == 1998 ~ 14,
            year == 2002 ~ 15,
            year == 2005 ~ 16,
            year == 2009 ~ 17,
            year == 2013 ~ 18,
            year == 2017 ~ 19,
            TRUE ~ NA_real_        
        )
    ) %>%
    dplyr::select(bundestag,party,seats, seats_total) %>% 
    group_by(bundestag, party) %>%
    summarise(seats = sum(seats),
              seats_share = seats/seats_total) %>%
    distinct(party, bundestag, seats, .keep_all = TRUE)


readr::write_csv(election_ger, here::here("data", "ParlGov", "election_germany.csv"))
