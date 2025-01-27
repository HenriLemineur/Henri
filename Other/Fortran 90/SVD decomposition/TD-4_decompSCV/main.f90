program main

implicit none

! Constantes de type explicitement déclarées
integer, parameter :: r = selected_real_kind(15, 307)
integer, parameter :: N = 3
integer :: M, i, LDVT, INFO, LWORK
real(r), dimension(:), allocatable :: x, y, S, WORK, c, SVD
real(r), dimension(:,:), allocatable :: A, U, VT, Sig, xy
integer, dimension(:), allocatable :: IWORK
character(128) :: tempFMT

! Lecture des données et allocation des tableaux
open(1, file = 'signal.dat')
    read(unit = 1, fmt = *)
    read(unit = 1, fmt = *) M
    
    LDVT = min(M, N)
    LWORK = 3 * min(M, N)**2 + max(max(M, N), 4 * min(M, N)**2 + 4 * min(M, N))
    allocate(x(M), y(M))
    allocate(A(M, N))
    allocate(S(min(M, N)), U(M, min(M, N)))
    allocate(VT(min(M, N), N))
    allocate(WORK(LWORK), IWORK(8 * min(M, N)))
    allocate(c(N))
    allocate(Sig(N, N))
    allocate(SVD(M))
    Sig = 0.0_r

    do i = 1, M
        read(unit = 1, fmt = *) x(i), y(i)
    end do
close(1)

! Calcul des phi_j(x_i)
A(:, 1) = phi1(x)
A(:, 2) = phi2(x)
A(:, 3) = phi3(x)

! Calcul de la décomposition en valeurs singulières (SVD)
call DGESDD('S', M, N, A, M, S, U, M, VT, LDVT, WORK, LWORK, IWORK, INFO)

if (INFO /= 0) then
    stop 'Erreur dans DGESDD'
end if

! Création de la matrice diag(sigma^-1)
do i = 1, min(M, N)
    if (S(i) /= 0.0_r) then
        Sig(i, i) = 1.0_r / S(i)
    else
        Sig(i, i) = 0.0_r
    end if
end do

! Calcul des coefficients c1, c2, c3
c = matmul(matmul(matmul(transpose(VT), Sig), transpose(U)), y)
print *, c

! Calcul des points interpolés par SVD
SVD = func(x, c(1), c(2), c(3))

! Formatage des résultats
allocate(xy(M, 2))
xy(:, 1) = x
xy(:, 2) = y

open(1, file = 'Initdat.dat')
    write(tempFMT, *) "(", M, "(e20.10,1x, e20.10, /))"
    write(1, tempFMT) transpose(xy)
close(1)

xy(:, 2) = SVD
open(1, file = 'Decompdat.dat')
    write(1, tempFMT) transpose(xy)
close(1)

contains

    elemental real(r) function phi1(x)
        real(r), intent(in) :: x
        phi1 = 1.0_r
    end function phi1

    elemental real(r) function phi2(x)
        real(r), intent(in) :: x
        phi2 = (x**2.0_r) * exp(-x)
    end function phi2

    elemental real(r) function phi3(x)
        real(r), intent(in) :: x
        phi3 = sin(x)**2.0_r
    end function phi3

    elemental real(r) function func(x, c1, c2, c3)
        real(r), intent(in) :: x, c1, c2, c3
        func = c1 * phi1(x) + c2 * phi2(x) + c3 * phi3(x)
    end function func

end program main
