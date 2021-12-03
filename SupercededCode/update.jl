#Bad code (idea was update parameters in simulation)
#=function for updating parameters
function update(sarsp,params)
	if sarsp.s′[end]==kickmade
		params.p_suc_kick=Beta(params.p_suc_kick.α+1,params.p_suc_kick.β)
		params.p_stop_kick=Beta(params.p_stop_kick.α,params.p_stop_kick.β)
	elseif sarsp.s′[end]==twomade
		params.p_suc_two=Beta(params_suc_two.α+1,params.p_suc_two.β)
		params.p_stop_two=Beta(params_stop_two.α,params.p_stop_two.β)
	elseif sarsp.s′[end]==missed && a==kick
		params.p_suc_kick=Beta(params.p_suc_kick.α,params.p_suc_kick.β+1)
		params.p_stop_kick=Beta(params.p_stop_kick.α+1,params.p_stop_kick.β)
	else
		params.p_suc_two=Beta(params.p_suc_two.α,params.p_suc_two.β+1)
		params.p_stop_two=Beta(params.p_stop_two.α+1,params.p_stop_two.β)
	end
end