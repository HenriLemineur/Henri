program Interpolation
  use nr, only: spline, splint, golden
  use donnees_communes, only: List_x, List_y, n, yp1, ypn, List_y2, max_points
  use outils, only: func
  implicit none

  ! Déclarations des variables
  integer :: i
  real(4) :: xmin, tol, a, b, c

  ! Initialiser tol
  tol = 1.0E-7_4

  ! Ouverture du fichier
  open(unit=10, file="cell.dat", status="old")

  ! Lire et ignorer la première ligne (en-tête)
  read(10, *)

  ! Lire la deuxième ligne pour obtenir le nombre de points
  read(10, *) n

  ! Allouer les tableaux List_x, List_y et List_y2
allocate(List_x(min(n, max_points)), List_y(min(n, max_points)), List_y2(min(n, max_points)))

  ! Lire les n lignes suivantes (courant et tension)
  do i = 1, n
    read(10, *) List_x(i), List_y(i)
  end do
  close(10)

  ! Affichage des données
  !print *, "Liste des courants (List_x) et tensions (List_y) :"
  !do i = 1, n
    !print "(F10.4, F10.4)", List_x(i), List_y(i)
  !end do

  ! Spline interpolation
  call spline(List_x, List_y, yp1, ypn, List_y2)

  ! Affichage des données après spline
  !print *, "List_y2"
  !do i = 1, n
    !print "(F10.4, F10.4)", List_y2(i)
  !end do

  ! Recherche du maximum de la fonction puissance
  a = List_x(1)
  b = List_x(n/2)
  c = List_x(n)

  xmin = golden(a, b, c, func, tol, xmin) 

  ! Affichage du courant maximisant la puissance
  print *, 'Le courant maximisant la puissance est ', abs(xmin)
end program Interpolation
