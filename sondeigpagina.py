import streamlit as st

def main():
    # Pone el título en la página
    st.title("Sondeig Interactivo")

    # Añade un campo de entrada de texto. Lo que el usuario escriba se guarda en la variable 'nombre'
    nombre = st.text_input("Por favor, introduce tu nombre")

    # Añade un botón. El código dentro del 'if' solo se ejecuta si se hace clic en el botón.
    if st.button("Saludar"):
        # Si el campo 'nombre' no está vacío...
        if nombre:
            st.success(f"¡Hola, {nombre}! Bienvenido a tu primera app interactiva.")
        # Si el campo 'nombre' está vacío...
        else:
            st.warning("Por favor, introduce un nombre antes de saludar.")

if __name__ == '__main__':
    main()