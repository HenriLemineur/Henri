program Integration
    use forsythe, only: quanc8
    use nrtype
    use modules, only: fun, open_file, close_file
    implicit none

    ! Déclaration des variables
    real(8) :: a, b, abserr, relerr, output, errest
    real(8) :: flag
    integer :: nofun, i
    integer, parameter :: n_cases = 16
    real(8), parameter :: r = 8
    real(8), dimension(3) :: relerr_cases = [1.0E-4_8, 1.0E-9_8, 1.0E-14_8]

    ! Bornes d'intégration
    a = 0.0_8
    b = 1.0_8

    ! Partie 1 : Intégration avec erreurs nulles
    abserr = 0.0_8
    relerr = 0.0_8

    call quanc8(fun, a, b, abserr, relerr, output, errest, nofun, flag)
    print *, "Resultats pour abserr = 0.0 et relerr = 0.0"
    print *, "Resultat :", output
    print *, "Erreur estimee :", errest
    print *, "Nombre d'evaluations :", nofun
    print *, "Flag :", flag
    print *, ""

    ! Partie 2 : Étude de la dépendance des erreurs relatives
    print *, "----------------------------------------------"
    print *, "Dependance en fonction de l'erreur relative"
    print *, "----------------------------------------------"
    print *, ""
    print *, "i | Result | Erreur estimee | Nbre d'evaluations"
    do i = 1, n_cases
        relerr = 10.0_8**(-i)
        abserr = 0.0_8
        call quanc8(fun, a, b, abserr, relerr, output, errest, nofun, flag)
        print *, i, output, errest, nofun
    end do
    print *, ""

    ! Partie 3 : Étude de la dépendance des erreurs absolues
    print *, "----------------------------------------------"
    print *, "Dependance en fonction de l'erreur absolue"
    print *, "----------------------------------------------"
    print *, ""
    print *, "i | Result | Erreur estimee | Nbre d'evaluations"
    do i = 1, n_cases
        abserr = 10.0_8**(-i)
        relerr = 0.0_8
        call quanc8(fun, a, b, abserr, relerr, output, errest, nofun, flag)
        print *, i, output, errest, nofun
    end do
    print *, ""

    ! Partie 4 : Visualisation des points d'évaluation
    print *, "----------------------------------------------"
    print *, "Visualisation des points d'evaluation"
    print *, "----------------------------------------------"
    print *, ""
    do i = 1, size(relerr_cases)
        relerr = relerr_cases(i)
        abserr = 0.0_8

        ! Ouvrir un fichier distinct pour chaque relerr avec un indice
        call open_file(i) 
        call quanc8(fun, a, b, abserr, relerr, output, errest, nofun, flag)
        call close_file()
    end do
end program Integration
