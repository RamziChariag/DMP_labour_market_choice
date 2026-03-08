module test
  use modeldef
  use params
  use stat_helper
  use solver_helper
  use glob
  use output
  use test_helper
contains

  subroutine testmatchingFunction(p)
    implicit none
    real(wp) :: x,y,z
    type (ExogenousParameters), intent(IN) :: p

    x = 0.9_wp
    y = 0.3_wp

    z=matchingFunction(x,y,p)
    write(*,*) 'res = ' ,z

    x = 0.1_wp
    y = 0.5_wp

    z=matchingFunction(x,y,p)
    write(*,*) 'res = ' ,z

  end subroutine testmatchingFunction

  subroutine testSmSpdHxy

    real(wp) :: st(5,6) , ht(5,6) , r(6)
    real(wp) :: s1 ,s2

    s1 = 1.1_wp
    s2 = 0.3_wp

    call generateArray(s1,st)
    call generateArray(s2,ht)
    call printMat(st)
    call printMat(ht)
    call IntSmSpdHxy(st, ht, r )
    call printVec(r)

  end subroutine testSmSpdHxy

  subroutine testIntSdVy

    real(wp) :: st(5,6) , vt(6) , r(5,6)
    real(wp) :: s1 ,s2

    s1 = 1.1_wp
    s2 = 0.3_wp

    call generateArray(s1,st)
    call generateVec(s2,vt)
    call printMat(st)
    call printVec(vt)
    call IntSdVy(st, vt, r )
    call printMat(r)

  end subroutine testIntSdVy

  subroutine testVM1

    real(wp) :: st(5,6) , vt(6) , r(5,6)
    real(wp) :: s1 ,s2

    s1 = 1.1_wp
    s2 = 0.3_wp

    call generateArray(s1,st)
    call generateVec(s2,vt)
    call printMat(st)
    call printVec(vt)
    call VM1(st, vt, r )
    call printMat(r)

  end subroutine testVM1

  subroutine testIntbarM1dHyp

    real(wp) :: st(5,6) , ht(5,6) , r(5,6)
    real(wp) :: s1 ,s2

    s1 = 1.1_wp
    s2 = 0.3_wp

    call generateArray(s1,st)
    call generateArray(s2,ht)
    call printMat(st)
    call printMat(ht)
    call IntbarM1dHyp(st, ht, r )
    call printMat(r)

  end subroutine testIntbarM1dHyp

  subroutine testcumsum
	implicit none
    real(wp) :: test_pdf(15), test_cdf(15)
    real(wp) :: s1 ,s2
	integer i

    s1 = 1.1_wp

    call generateVec(s1,test_pdf)
    test_pdf = abs(test_pdf)
    test_pdf = test_pdf / sum(test_pdf)

	call pdf2cdf(test_pdf,test_cdf)

	call printVec(test_cdf)

	call locate(test_cdf,0.2_wp,i)
	write(*,*) i

  end subroutine testcumsum


end module test


