mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"harison.pame@outlook.fr\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml