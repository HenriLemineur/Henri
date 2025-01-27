program main
    use forsythe, only: decomp, solve
    use matrix_module
    implicit none

    integer, parameter :: r = 8
    integer :: N, i
    real(r), dimension(:,:), allocatable :: M
    real(r), dimension(:), allocatable :: Vmat, Imat, mywork
    integer, dimension(:), allocatable :: myipvt
    real(r) :: condition, ResEquiv
    real(r), parameter :: Vval = 1.0_r  ! Tension d'entrée (constante)

    ! Lecture de la taille du circuit
    print *, "Entrez le nombre de boucles N (>= 1) :"
    read *, N

    ! Allocation des matrices et vecteurs
    allocate(M(N+1, N+1))
    allocate(Vmat(N+1))
    allocate(Imat(N+1))
    allocate(mywork(N+1))
    allocate(myipvt(N+1))

    ! Appel de la subroutine pour créer la matrice
    call create_matrix(N, M)

    ! Initialisation du vecteur des tensions Vmat
    Vmat = 0.0_r
    Vmat(1) = Vval

    ! Triangularisation de la matrice M
    call decomp(N+1, N+1, M, condition, myipvt, mywork)
    print *, "Condition estimée de la matrice M :", condition

    ! Résolution du système M * I = Vmat pour obtenir les courants
    call solve(N+1, N+1, M, Vmat, myipvt)

    ! Calcul des courants
    Imat = Vmat

    ! Calcul de la résistance équivalente
    ResEquiv = Vval / Imat(1)

    ! Affichage des résultats
    print *, "Courants dans les boucles :"
    do i = 1, N+1
        print '(A, I0, A, F8.5)', 'I(', i, ') = ', Imat(i)
    end do


    ! Libération de la mémoire
    deallocate(M, Vmat, Imat, mywork, myipvt)
end program main
