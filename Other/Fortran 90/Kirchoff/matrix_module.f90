module matrix_module
  implicit none
contains
  subroutine create_matrix(N, matrix)
    integer, intent(in) :: N
    real(8), dimension(:,:), allocatable, intent(out) :: matrix
    integer :: i, halfN

    ! Allocation de la matrice (N+1)x(N+1)
    allocate(matrix(N+1, N+1))
    
    ! Calcul de la partie entière de N/2
    halfN = N / 2

    ! Initialisation de la matrice à zéro
    matrix = 0.0d0  ! Initialiser tous les éléments à zéro

    ! Placer les -1 dans les trois dernières positions de la première ligne et de la première colonne
    do i = N+2-halfN, N+1
        matrix(1, i) = -1.0d0   ! Placer les 3 derniers -1 dans la première ligne
        matrix(i, 1) = -1.0d0   ! Placer les 3 derniers -1 dans la première colonne
    end do

    ! Ajouter -1 aux éléments (N+1, 2) et (2, N+1)
    matrix(N+1, 2) = -1.0d0
    matrix(2, N+1) = -1.0d0

    ! Remplir la diagonale principale avec floor(N/2) pour tous les éléments de [2,2] à [N+1,N+1]
    matrix(2, 2) = real(halfN, 8)  ! L'élément (2,2) est floor(N/2), convertir en REAL(8)
    do i = 3, N+1
        matrix(i, i) = real(halfN, 8)  ! Remplir la diagonale avec floor(N/2), convertir en REAL(8)
    end do

    ! Ajouter les -1 autour des éléments de la diagonale principale (hors première ligne/colonne)
    do i = 2, N
        matrix(i-1, i) = -1.0d0    ! au-dessus de la diagonale
        matrix(i+1, i) = -1.0d0    ! en-dessous de la diagonale
        matrix(i, i-1) = -1.0d0    ! à gauche de la diagonale
        matrix(i, i+1) = -1.0d0    ! à droite de la diagonale
    end do

    ! Remplir le dernier élément (N+1, N+1) avec floor(N/2)
    matrix(N+1, N+1) = real(halfN, 8)

    ! Remplir la première ligne et la première colonne
    matrix(1, 1) = real(halfN, 8)   ! L'élément (1,1) est la partie entière de N/2
    matrix(1, 2) = 0.0d0       ! L'élément (1,2) est 0
    matrix(2, 1) = 0.0d0       ! L'élément (2,1) est 0
  end subroutine create_matrix
end module matrix_module
