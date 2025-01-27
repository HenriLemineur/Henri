module modules
    use nrtype
    implicit none

    integer :: file_unit
    logical :: file_opened = .false.

contains

    real(8) function fun(x)
        real(8), intent(in) :: x

        ! Calcul de la fonction
        fun = 1.0_8 / ((x - 0.3_8)**2 + 0.01_8) + & 
              1.0_8 / ((x - 0.9_8)**2 + 0.04_8) - 6.0_8

        ! Écriture dans le fichier si ouvert
        if (file_opened) then
            write(file_unit, '(F10.5, 1X, F10.5)') x, fun
        end if
    end function fun

    subroutine open_file(index)
        integer, intent(in) :: index
        character(len=50) :: filename

        ! Création d'un fichier avec un indice
        write(filename, '("points", I1, ".dat")') index
        open(unit=file_unit, file=trim(adjustl(filename)), status="replace", action="write")
        file_opened = .true.
    end subroutine open_file

    subroutine close_file()
        ! Fermer le fichier après utilisation
        if (file_opened) then
            close(unit=file_unit)
            file_opened = .false.
        end if
    end subroutine close_file

end module modules
