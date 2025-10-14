import pickle
import streamlit as st
import chess
import chess.svg
import base64

def render_svg(svg_data):
    """Показывает SVG доску в Streamlit."""
    b64 = base64.b64encode(svg_data.encode("utf-8")).decode("utf-8")
    html = f'<img src="data:image/svg+xml;base64,{b64}"/>'
    st.markdown(html, unsafe_allow_html=True)

def main():
    st.title("♟️ Game Viewer — AlphaZero Chess")

    uploaded_file = st.file_uploader("Выбери .pkl файл партии", type="pkl")

    if uploaded_file is not None:
        game_info = pickle.load(uploaded_file)
        moves = game_info.get("moves", [])

        st.write(f"**{game_info['title']}**")
        st.write(f"Победитель: {game_info['winner']}")
        st.write(f"**Ходов в партии:** {len(moves)}")

        # инициализация состояния
        if "move_idx" not in st.session_state:
            st.session_state.move_idx = 0

        # кнопки навигации
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("⏮️ В начало"):
                st.session_state.move_idx = 0
        with col2:
            if st.button("⬅️ Назад"):
                if st.session_state.move_idx > 0:
                    st.session_state.move_idx -= 1
        with col3:
            if st.button("Вперёд ➡️"):
                if st.session_state.move_idx < len(moves):
                    st.session_state.move_idx += 1
        with col4:
            if st.button("⏭️ В конец"):
                st.session_state.move_idx = len(moves)

        st.write(f"**Текущий ход:** {st.session_state.move_idx}/{len(moves)}")

        # восстановление доски
        board = chess.Board()
        for m in moves[:st.session_state.move_idx]:
            board.push_uci(m)

        # отображение доски
        svg_data = chess.svg.board(board)
        render_svg(svg_data)

if __name__ == "__main__":
    main()
#streamlit run src/view_game.py
#http://localhost:8501