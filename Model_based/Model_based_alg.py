import torch
class MB_alg:

    def __init__(self,est_model,actor,t_step,n_parametrised_steps, velocity_w, th_error):

        self.est_model = est_model
        self.actor = actor
        self.n_parametrised_steps = n_parametrised_steps
        self.velocity_weight = velocity_w

        self.t_step = t_step

        self.ModelError_th = 0.001#0.0001#0.015
        self.th_error = th_error

    def update_model(self, actions, trg_y,target_arm):

        trg_y = trg_y.squeeze()
        est_y = self.est_model.perform_reaching(self.t_step,actions).squeeze()
        ep = 0

        ep_acc = []
        ep_loss = []
        max_error = torch.max(torch.abs(trg_y - est_y))

        trg_alpha = target_arm.alpha
        trg_omega = target_arm.omega

        while max_error > self.ModelError_th:

           model_loss = self.est_model.update(trg_y, est_y)
           est_y = self.est_model.perform_reaching(self.t_step,actions).squeeze()
           ep_loss.append(model_loss)
           max_error = torch.max(torch.abs(trg_y - est_y))
           ep_acc.append(max_error)
           ep += 1

           if ep % 50 == 0:

               avr_error = sum(ep_acc) /50
               avr_loss = sum(ep_loss)/50

               print("Model upd ep: ",ep)
               print("avr error: ", avr_error)
               print("avr loss: ", avr_loss)
               print("alpha difference: ", trg_alpha - self.est_model.alpha)
               print("omega difference: ", trg_omega - self.est_model.omega, "\n")
               ep_acc = []
               ep_loss = []


        print(self.est_model.alpha)
        print(self.est_model.omega)
        print(self.est_model.F,"\n")
        return ep


    def update_actor(self, target_st):

     ep = 0
     ep_distance = []
     ep_velocity = []
     print_rwd = 50 # initialise to high random value

     while print_rwd > self.th_error:

            ep += 1

            actions = self.actor(target_st).view(1,2,self.n_parametrised_steps)

            thetas = self.est_model.perform_reaching(self.t_step, actions)

            rwd = self.est_model.compute_rwd(thetas, target_st[0,0], target_st[0,1], -1)
            velocity = self.est_model.compute_vel(thetas, -1)
            loss = rwd + (velocity * self.velocity_weight)

            self.actor.update(loss)

            ep_distance.append(torch.sqrt(rwd).detach())  # mean distance to assess performance
            ep_velocity.append(torch.sqrt(velocity).detach())


            if ep % 100 == 0:

                print_rwd = sum(ep_distance) / 100
                print_velocity = sum(ep_velocity) / 100

                print("Actor update ep: ", ep)
                print("Accuracy: ", print_rwd)
                print("Velocity: ", print_velocity, "\n")
                ep_distance = []
                ep_velocity = []